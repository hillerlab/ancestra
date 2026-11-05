# Vendored from mamba_ssm==2.3.2.post1; source SHA-256: addcf96f306dd2c9bf1c9ef9feda42381a6e55ad906dd92e220598fcfec7b508; imports: mamba_ssm.* -> ancestra._vendor.mamba_ssm.*.
import triton
import triton.language as tl
from packaging import version

TRITON3 = version.parse(triton.__version__) >= version.parse("3.0.0")


if TRITON3:
    @triton.jit
    def softplus(dt):
        return tl.math.log(tl.math.exp(dt) + 1)
else:
    @triton.jit
    def softplus(dt):
        return tl.math.log1p(tl.exp(dt))