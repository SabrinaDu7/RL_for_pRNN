"""The two learned networks, and the device machinery they share.

  policy.py        ACModel / ACModelSR - the actor-critic. ACModelSR is the one
                   that takes the pRNN's hidden state as input, which is how the
                   policy sees the world model.
  prnn_adapter.py  PRNNAdapter - the world model seam. THE ONLY MODULE IN THIS
                   PACKAGE THAT IMPORTS `prnn`; everything else talks to it.
  device.py        on_device / eval_mode - moving these models between CPU and
                   accelerator without changing their identity.

`device.py` is here because it operates ON models, not because it is one. Its
address-preserving contract is load-bearing: the spatial evaluation runs on CPU,
and a naive `.to()` there silently invalidated captured CUDA graphs.
"""
