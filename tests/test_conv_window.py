#!/usr/bin/env python3
"""CPU check: a 2D convolution whose input is exactly one kernel window equals the matmul that replaces it."""
import unittest

import torch
import torch.nn.functional as F

from exllamav3.modules.conv import conv2d_one_window


class ConvWindowTests(unittest.TestCase):
    def test_matches_conv2d_for_every_batch_size(self):
        torch.manual_seed(0)
        c_in, c_out, k = 24, 40, 2
        weight = torch.randn(c_out, c_in, k, k)
        bias = torch.randn(c_out)
        for n in (1, 7, 238, 1500):
            x = torch.randn(n, c_in, k, k)
            ref = F.conv2d(x, weight, bias, (k, k)).flatten(2).permute(0, 2, 1)
            got = conv2d_one_window(x, weight, bias)
            self.assertEqual(tuple(got.shape), (n, 1, c_out))
            torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)

    def test_without_bias(self):
        weight = torch.randn(5, 3, 2, 2)
        x = torch.randn(9, 3, 2, 2)
        ref = F.conv2d(x, weight, None, (2, 2)).flatten(2).permute(0, 2, 1)
        torch.testing.assert_close(conv2d_one_window(x, weight, None), ref, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
