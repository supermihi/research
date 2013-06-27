#!/usr/bin/python2
import numpy as np
cimport numpy as np
fpt = np.int64
ctypedef np.int64_t fpt_t

cdef:
    int _frac, _int, _total
    double _scale, _revscale
    fpt_t _mask, _fracMask, _sign, _carry, _negExtension

_fmt = None

def setPrecision(int total, int fractional):
    global _frac, _int, _total, _scale, _revscale, _fmt, _mask, _fracMask, _sign, _carry, _negExtension
    _int = total - fractional
    _frac = fractional
    _total = total
    _scale = .5**fractional
    _revscale = 2**fractional
    _fmt = "{0:0"+str(_int)+"b}.{1:0"+str(_frac)+"b}"
    _mask = 2**total-1
    _fracMask = 2**_frac-1
    _sign = fpt(1) << (_total-1)
    _carry = fpt(1) << _total
    _negExtension = -1 ^ _mask
setPrecision(6, 2)

def float2fixed(value):
    if value < 0:
        fixed = _carry+fpt(np.floor(value*_revscale))
        if not (fixed & _sign) or fixed >> _total:
            raise ValueError("Underflow: {} too small".format(value))
        fixed |= _negExtension
    else:
        fixed = fpt(value*_revscale)
        if fixed >> (_total-1):
            raise ValueError("Overflow: {} too big".format(value))
    return fixed
    
def fixed2float(value):
    result =  (value & _mask)*_scale
    if value & _sign:
        return result-2**_int
    return result
    
def strBits(value):
    return _fmt.format((value & _mask)>>_frac, value & (2**_frac-1))

def printBits(value):
    print(strBits(value))

def signExtend(value):
    if value & _sign:
        return _negExtension | value
    return value
    
cdef inline bint isSignExtended(fpt_t value):
    if value & _sign:
        return (value & _negExtension) == _negExtension
    else:
        return (value & _negExtension) == 0

cpdef fpt_t add(fpt_t f1, fpt_t f2):
    cdef fpt_t result = f2 + f1
    if bool(result & _sign) ^ bool(result & _carry):
        raise ValueError("Addition overflow")
    return result

def negate(f1):
    result = fpt(1<<_total)-f1
    if result == 1<<(_total-1):
        raise ValueError("Negation overflow")
    return result & _mask

def sub(f1, f2):
    return add(f1, negate(f2))
    
cpdef fpt_t mul(fpt_t f1, fpt_t f2):
    cdef fpt_t result = f1 * f2
    if result & _fracMask:
        print('warning: multiplication precision loss')
    result >>= _frac
    if not isSignExtended(result):
        raise ValueError("Multiplication overflow")
    return result
    
    
def div(f1, f2):
    result, remainder = divmod((signExtend(f1)<<_frac), signExtend(f2))
    if not isSignExtended(result):
        raise ValueError("Division overflow")
    if remainder != 0:
        print("warning: division precision loss")
    if result < 0:
        result = negate(-result)
    return result

import numbers
class FixedPointNumber:
    
    def __init__(self, floatValue=None, fpValue=None):
        if floatValue is not None:
            self.value = float2fixed(floatValue)
        elif fpValue is not None:
            self.value = fpValue
        else:
            raise ValueError("Need to supply at least floatValue or fpValue")
        
    def __add__(self, other):
        if not isinstance(other, FixedPointNumber):
            raise NotImplementedError()
        return FixedPointNumber(fpValue=add(self.value, other.value))
        
    def __iadd__(self, other):
        if not isinstance(other, FixedPointNumber):
            raise NotImplementedError()
        self.value = add(self.value, other.value)
        return self
    
    def __sub__(self, other):
        if not isinstance(other, FixedPointNumber):
            raise NotImplementedError()
        return FixedPointNumber(fpValue=sub(self.value, other.value))
    
    def __isub__(self, other):
        if not isinstance(other, FixedPointNumber):
            raise NotImplementedError()
        self.value = sub(self.value, other.value)
        return self
    
    def __mul__(self, other):
        if not isinstance(other, FixedPointNumber):
            raise NotImplementedError()
        return FixedPointNumber(fpValue=mul(self.value, other.value))
    
    def __imul__(self, other):
        if not isinstance(other, FixedPointNumber):
            raise NotImplementedError()
        self.value = mul(self.value, other.value)
        return self
    
    def __div__(self, other):
        if not isinstance(other, FixedPointNumber):
            raise NotImplementedError()
        return FixedPointNumber(fpValue=div(self.value, other.value))

    def __idiv__(self, other):
        if not isinstance(other, FixedPointNumber):
            raise NotImplementedError()
        self.value = div(self.value, other.value)
        return self
        
    def __neg__(self):
        return FixedPointNumber(fpValue=negate(self.value))
    
    def __repr__(self):
        return strBits(self.value)
        
    def __str__(self):
        return str(fixed2float(self.value))
