"""Torch autodifferentiation for MRI-NUFFT."""

import torch


# class _NUFFT_OP(torch.autograd.Function):
#     """Autograd support for op nufft function."""

#     @staticmethod
#     def forward(ctx, x, nufft_op):
#         """Forward image -> k-space."""
#         ctx.save_for_backward(x)
#         ctx.nufft_op = nufft_op
#         return nufft_op.op(x)

#     @staticmethod
#     def backward(ctx, dy):
#         """Backward image -> k-space."""
#         (x,) = ctx.saved_tensors

#         return ctx.nufft_op.adj_op(dy),  None 
        
class _NUFFT_OP(torch.autograd.Function):
    """Autograd support for op nufft function."""

    @staticmethod
    def forward(ctx, x, nufft_op, traj):
        """Forward image -> k-space."""
        ctx.save_for_backward(x, traj)
        ctx.nufft_op = nufft_op
        return nufft_op.op(x)
    

    @staticmethod
    def backward(ctx, dy):
        """Backward image -> k-space."""
        print(dy.shape)
        (x,traj) = ctx.saved_tensors 

        im_size = x.size()[1:] #[16, 16]
        r  = [torch.linspace(-size / 2, size / 2 - 1, size) for size in im_size] #len(r) = 2 / [16] for each
        grid_r = torch.meshgrid(*r, indexing='ij')
        grid_r = torch.stack(grid_r, dim=0).type_as(x)[None, ...]# add batch size [1, 2, 16, 16]

        grid_x = x * grid_r
        nufft_dx_dom = torch.cat([ctx.nufft_op.op(grid_x[:, i:i+1, :, :]) for i in range(grid_x.size(1))], dim=1)
        #nufft_dx_dom = ctx.nufft_op.op(x * grid_r) # not work beacuse op only accpect [1, 1, *im_size]
        
        grad_traj = torch.transpose((-1j * torch.conj(dy) * nufft_dx_dom).squeeze(), 0, 1).type_as(traj) #dy should be [1, 1, 256] nufft_dx_dom should be [1, 2, 256] the first dim is batch size which should be reserved for the nufft
    


        return ctx.nufft_op.adj_op(dy), None, grad_traj     


class _NUFFT_ADJOP(torch.autograd.Function):
    """Autograd support for adj_op nufft function."""

    @staticmethod
    def forward(ctx, y, nufft_op):
        """Forward kspace -> image."""
        ctx.save_for_backward(y)
        ctx.nufft_op = nufft_op
        return nufft_op.adj_op(y)

    @staticmethod
    def backward(ctx, dx):
        """Backward kspace -> image."""
        (y,) = ctx.saved_tensors
        return ctx.nufft_op.op(dx), None


class MRINufftAutoGrad(torch.nn.Module):
    """
    Wraps the NUFFT operator to support torch autodiff.

    Parameters
    ----------
    nufft_op: Classic Non differentiable MRI-NUFFT operator.
    """

    def __init__(self, nufft_op):
        super().__init__()
        if nufft_op.squeeze_dims:
            raise ValueError("Squeezing dimensions is not " "supported for autodiff.")
        self.nufft_op = nufft_op

    def op(self, x, traj):
        r"""Compute the forward image -> k-space."""
        return _NUFFT_OP.apply(x, self.nufft_op, traj)

    def adj_op(self, kspace):
        r"""Compute the adjoint k-space -> image."""
        return _NUFFT_ADJOP.apply(kspace, self.nufft_op)

    def __getattr__(self, name):
        """Get the attribute from the root operator."""
        return getattr(self.nufft_op, name)
