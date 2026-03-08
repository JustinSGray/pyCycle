# Backward compatibility - moved to pycycle.new_elements.jax_element_base
from pycycle.new_elements.jax_element_base import (
    JaxElement, reset_timing_stats, print_timing_stats,
    get_timing_stats, clear_jit_cache, _jax_element_timing_stats,
)
