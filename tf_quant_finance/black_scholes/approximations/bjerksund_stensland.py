# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Bjerksund-Stensland price approximation of a batch of American options."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp


def _phi(S, T, gamma, H, X, vols, r, b, dtype):
    lambd = (-r + gamma * b + 0.5 * gamma * (gamma - 1) * vols) * T
    d_denom = -tf.log(S / H) + (b + (gamma - 0.5) * vols) * T
    d = d_denom / tf.sqrt(vols * T)
    d2 = d - 2 * tf.math.log(X/S) / tf.sqrt(vols * T)
    kappa = 2 * b / vols + 2 * gamma - 1
    normal = tfp.distributions.Normal(
        loc=tf.zeros([], dtype=dtype), scale=1)
    phi_int = normal.cdf(d) - tf.pow(X/S, kappa) * normal.cdf(d2)
    return tf.mult(tf.exp(lambd) * tf.pow(S, gamma),
                   phi_int,
                   name='Bjerksund-Stensland phi')


def _boundary(strikes, vols, expiries, r, b, beta):
    """
    Bjerksund-Stensland exercise boundary
    """
    b0 = tf.max(strikes, r / (r-b) * strikes)
    binf = beta / (beta - 1) * strikes
    h = -((b * expiries + 2 * vols * tf.sqrt(expiries))
          * b0 / (b0 - binf))
    return b0 + (binf - b0) * (1 - tf.exp(h))


def option_price(forwards,
                 strikes,
                 volatilities,
                 expiries,
                 dividends=None,
                 discount_factors=None,
                 is_call_options=None,
                 dtype=None,
                 name=None):
    """Computes the Bkerksund-Stensland price for a batch of American options.

  ## References:
XXX  [1] Hull, John C., Options, Futures and Other Derivatives. Pearson, 2018.
XXX  [2] Wikipedia contributors. Black-Scholes model. Available at:
    https://en.wikipedia.org/w/index.php?title=Black%E2%80%93Scholes_model

  Args:
    forwards: A real `Tensor` of any shape. The current forward prices to
      expiry.
    strikes: A real `Tensor` of the same shape and dtype as `forwards`. The
      strikes of the options to be priced.
    volatilities: A real `Tensor` of same shape and dtype as `forwards`. The
      volatility to expiry.
    expiries: A real `Tensor` of same shape and dtype as `forwards`. The expiry
      for each option. The units should be such that `expiry * volatility**2` is
      dimensionless.
    discount_factors: A real `Tensor` of same shape and dtype as the `forwards`.
      The discount factors to expiry (i.e. e^(-rT)). If not specified, no
      discounting is applied (i.e. the undiscounted option price is returned).
      Default value: None, interpreted as discount factors = 1.
    is_call_options: A boolean `Tensor` of a shape compatible with `forwards`.
      Indicates whether to compute the price of a call (if True) or a put (if
      False). If not supplied, it is assumed that every element is a call.
    dtype: Optional `tf.DType`. If supplied, the dtype to be used for conversion
      of any supplied non-`Tensor` arguments to `Tensor`.
      Default value: None which maps to the default dtype inferred by TensorFlow
      (float32).
    name: str. The name for the ops created by this function.
      Default value: None which is mapped to the default name `option_price`.

  Returns:
    option_prices: A `Tensor` of the same shape as `forwards`. The Black
    Scholes price of the options.


"""
    forwards = tf.convert_to_tensor(forwards, dtype=dtype, name='forwards')
    strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
    volatilities = tf.convert_to_tensor(
        volatilities, dtype=dtype, name='volatilities')
    expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
    if is_call_options is None:
        is_call_options = True
    if dividends is None:
        dividends = 0
    if discount_factors is None:
        discount_factors = 1
    dividends = tf.convert_to_tensor(dividends, dtype=dtype, name='dividend')
    discount_factors = tf.convert_to_tensor(
        discount_factors, dtype=dtype, name='discount_factors')
    # convert puts to calls using American option style put-call parity
    forwards_for_calls = tf.where_v2(is_call_options, forwards, strikes)
    strikes_for_calls = tf.where_v2(is_call_options, strikes, forwards)
    dividend_for_calls = tf.where_v2(is_call_options,
                                     dividends,
                                     discount_factors)
    discount_for_calls = tf.where_v2(is_call_options,
                                     discount_factors,
                                     dividends)
    # where dividend is zero - use Merton's no-early exercise theorem

    # use shorthands for variables in formulas
    S = forwards_for_calls
    K = strikes_for_calls
    T = expiries
    q = dividend_for_calls
    r = discount_for_calls
    b = discount_for_calls - dividend_for_calls
    beta0 = (0.5 - b / volatilities)
    beta = beta0 + tf.sqrt(tf.square(beta0)
                           + 2 * r / volatilities)
    boundary = _boundary(S, volatilities, T, r, b, beta)
    term1 = ((boundary - K)
            * tf.pow(K / boundary, beta)
            * (1 - _phi(S, T, beta, boundary, boundary,
                        r, b, volatilities, dtype)))
    term2 = S * _phi(S, T, 1.0, boundary, boundary, r, b, volatilities, dtype)
    term3 = S * _phi(S, T, 1.0, K, boundary, r, b, volatilities, dtype)
    term4 = K * _phi(S, T, 0, boundary, boundary, r, b, volatilities, dtype)
    term5 = K * _phi(S, T, 0, boundary, boundary, r, b, volatilities, dtype)
    return tf.add(term1, term2 - term3 -term4 + term5, name = name)
