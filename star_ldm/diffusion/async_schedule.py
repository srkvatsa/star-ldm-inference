
import torch
from typing import Callable, Optional, Tuple

class AsyncDiffusionScheduler:

    def __init__(
        self,
        noise_schedule_fn: Callable,
        device: str = 'mps',
        z_shape: Optional[Tuple[int, ...]] = None,
    ):
        self.noise_schedule_fn = noise_schedule_fn
        self.device = torch.device(device)
        self.z_shape = z_shape

        self._cpu_noise: Optional[torch.Tensor] = None
        self._precomputed_valid = False

    def precompute_noise(self, z_shape: Tuple[int, ...]) -> None:
        self.z_shape = z_shape

        self._cpu_noise = torch.randn(z_shape, device='cpu', dtype=torch.float32)
        self._precomputed_valid = True

    def get_noise(self) -> torch.Tensor:
        if self._precomputed_valid and self._cpu_noise is not None:
            noise = self._cpu_noise.to(self.device)
            self._precomputed_valid = False
            return noise
        else:

            if self.z_shape is not None:
                return torch.randn(self.z_shape, device=self.device)
            raise RuntimeError("No precomputed noise available and z_shape not set")

    def compute_schedule_values(
        self, time: torch.Tensor, time_next: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        with torch.no_grad():
            time_cpu = time.cpu() if time.device.type != 'cpu' else time
            time_next_cpu = time_next.cpu() if time_next.device.type != 'cpu' else time_next

            alpha2_cpu = self.noise_schedule_fn(time_cpu).unsqueeze(-1)
            alpha2_next_cpu = self.noise_schedule_fn(time_next_cpu).unsqueeze(-1)

        alpha2 = alpha2_cpu.to(self.device)
        alpha2_next = alpha2_next_cpu.to(self.device)

        return alpha2, alpha2_next

def create_async_diffusion_loop(
    model_forward_fn: Callable,
    noise_schedule_fn: Callable,
    ddpm_step_fn: Callable,
    time_pairs: list,
    z_t: torch.Tensor,
    device: str = 'mps',
    var_lambda: float = 0.2,
    **model_kwargs,
) -> torch.Tensor:
    scheduler = AsyncDiffusionScheduler(noise_schedule_fn, device, z_t.shape)
    x_start = None

    scheduler.precompute_noise(z_t.shape)

    for i, (time, time_next) in enumerate(time_pairs):

        alpha2, alpha2_next = scheduler.compute_schedule_values(time, time_next)

        if i + 1 < len(time_pairs):

            import threading
            noise_thread = threading.Thread(
                target=scheduler.precompute_noise,
                args=(z_t.shape,)
            )
            noise_thread.start()

        model_output = model_forward_fn(z_t, alpha2, **model_kwargs)
        x_start = model_output.pred_x
        eps = model_output.pred_eps

        if time_next[0] <= 0:
            z_t = x_start

            if i + 1 < len(time_pairs) and noise_thread is not None:
                noise_thread.join()
            continue

        if i + 1 < len(time_pairs):
            noise_thread.join()

        noise = scheduler.get_noise()

        z_t = ddpm_step_fn(z_t, eps, noise, alpha2, alpha2_next, var_lambda)

        if i + 2 < len(time_pairs):
            pass

    return x_start
