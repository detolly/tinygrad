from enum import Enum
from typing import List, NamedTuple, Optional, Tuple, Union, cast
from typing_extensions import Dict
from tinygrad.dtype import DType, PtrDType, dtypes
from tinygrad.ops import BinaryOps, PatternMatcher, UOp, UOps, UPat
from tinygrad.renderer import Renderer
import platform, os

def little_endian(i: int) -> list[int]: return [i & 0xFF, (i & 0xFF00) >> 8, (i & 0xFF0000) >> 16, (i & 0xFF000000) >> 24]

class X86R(Enum):
  RAX = 0; RCX = 1; RDX = 2; RBX = 3; RBP = 5; RSI = 6; RDI = 7
  R8 = 8; R9 = 9; R10 = 10; R11 = 11; R12 = 12; R13 = 13; R14 = 14; R15 = 15
class X86FR(Enum):
  XMM0 = 0; XMM1 = 1; XMM2 = 2; XMM3 = 3; XMM4 = 4; XMM5 = 5; XMM6 = 6; XMM7 = 7; XMM8 = 8
  XMM9 = 9; XMM10 = 10; XMM11 = 11; XMM12 = 12; XMM13 = 13; XMM14 = 14; XMM15 = 15;
  YMM0 = 0; YMM1 = 1; YMM2 = 2; YMM3 = 3; YMM4 = 4; YMM5 = 5; YMM6 = 6; YMM7 = 7; YMM8 = 8
  YMM9 = 9; YMM10 = 10; YMM11 = 11; YMM12 = 12; YMM13 = 13; YMM14 = 14; YMM15 = 15;
  ZMM0 = 0; ZMM1 = 1; ZMM2 = 2; ZMM3 = 3; ZMM4 = 4; ZMM5 = 5; ZMM6 = 6; ZMM7 = 7; ZMM8 = 8
  ZMM9 = 9; ZMM10 = 10; ZMM11 = 11; ZMM12 = 12; ZMM13 = 13; ZMM14 = 14; ZMM15 = 15;

X86_REGISTERS_THAT_NEED_TO_BE_SAVED = [X86R.RBX, X86R.R12, X86R.R13, X86R.R14, X86R.R15]
X86_ARG_REGISTERS = [X86R.RDI, X86R.RSI, X86R.RDX, X86R.RCX, X86R.R8, X86R.R9]

X86_64BIT_PREFIX = 0b01001000; 
X86_MODRM = 0b11000000; 
X86_REX_W = 1 << 3; X86_REX_R = 1 << 2; X86_REX_X = 1 << 1; X86_REX_B = 1;

def prefix_byte_reg(dest_register: X86R): return X86_64BIT_PREFIX | (X86_REX_R if dest_register.value >= X86R.R8.value else 0)
def prefix_byte_rm(source_register: X86R): return X86_64BIT_PREFIX | (X86_REX_B if source_register.value >= X86R.R8.value else 0)
def modrm_byte_reg(reg: X86R): return X86_MODRM | ((reg.value & 0b111) << 3);
def modrm_byte_rm(rm: X86R): return X86_MODRM | (rm.value & 0b111);
def modrm_byte_reg_opcode(opcode: int): return X86_MODRM | ((opcode & 0b111) << 3);

def modrm_reg_to_reg(rm: X86R, reg: X86R): return modrm_byte_rm(rm) | modrm_byte_reg(reg)
def prefix_reg_to_reg(rm: X86R, reg: X86R): return prefix_byte_rm(rm) | prefix_byte_reg(reg)

class X8664ASMRenderer(Renderer):
  def __init__(self, avx=False, avx2=False, avx512=False): self.avx, self.avx2, self.avx512 = avx, avx2, avx512; self.reset()
  def alloc_register(self, uop: Optional[UOp]=None) -> X86R: 
    reg = filter(lambda r: self.free_registers[r], X86R).__next__()
    if uop is not None:
      self.uop_registers[uop] = reg
    self.free_registers[reg] = False
    return reg
  def dealloc_register(self, reg): assert not self.free_registers[reg]; self.free_registers[reg] = True
  def alloc_fpregister(self) -> X86FR:
    reg = filter(lambda r: self.free_fpu_registers[r], X86FR).__next__()
    self.free_fpu_registers[reg] = False
    return reg
  def dealloc_fpregister(self, reg): assert not self.free_fpu_registers[reg]; self.free_fpu_registers[reg] = True
  def alloc_argument_register(self, uop: UOp): 
    if isinstance(uop.dtype, PtrDType):
      reg = filter(lambda r: self.free_registers[r], X86_ARG_REGISTERS).__next__()
      self.free_registers[reg] = False; self.buffer_registers[uop.arg] = (reg, uop.dtype); self.uop_registers[uop] = reg
      return reg
    assert False, "something that wasnt a buffer in arguments"

  def emit(self, code: list[int]): self.code = self.code + bytearray(code)
  def tell(self): return len(self.code) - 1

  def xor(self, rm: X86R, reg: X86R): self.emit([prefix_reg_to_reg(rm, reg), 0x31, modrm_reg_to_reg(rm, reg)])
  def jne(self, rip_relative_offset: int): self.emit([0x0F, 0x85] + little_endian(rip_relative_offset-0x7))
  def sub(self, op1: X86R, immediate32: int): self.emit([prefix_byte_rm(op1) | X86_REX_W, 0x81, modrm_byte_rm(op1) | modrm_byte_reg_opcode(5)] + little_endian(immediate32))
  def mov(self, op1: X86R, immediate32: int): self.emit([prefix_byte_rm(op1) | X86_REX_W, 0xC7, modrm_byte_rm(op1) | modrm_byte_reg_opcode(0)] + little_endian(immediate32))

  def reset(self):
    self.free_registers = { r: True for r in X86R }
    self.free_fpu_registers = { r: True for r in X86FR }
    self.code = bytearray()
    self.buffer_registers: Dict[int, Tuple[X86R, DType]] = {}
    self.uop_registers: Dict[UOp, X86R] = {}

  def render_recursive(self, uops: List[UOp]) -> int:
    i, max = 0, len(uops)
    while i < max:
      u = uops[i]
      uop,dtype,src,args = u.op,u.dtype,u.src,u.arg
      print(u.render(False))

      if uop == UOps.DEFINE_GLOBAL: self.alloc_argument_register(u)
      elif uop == UOps.CONST: pass
      elif uop == UOps.RANGE: 
        counter_register = self.alloc_register(u)
        self.mov(counter_register, src[1].arg)
        assert src[0].arg == 0
        loop_begin = self.tell()
        i += self.render_recursive(uops[i+1:])
        self.sub(counter_register, 1)
        self.jne(-(self.tell() - loop_begin))
        self.dealloc_register(counter_register)
      elif uop == UOps.ENDRANGE: return i + 1
      elif uop == UOps.STORE:
        assert src[0].op == UOps.DEFINE_GLOBAL
        if src[2].op == UOps.CONST or src[2].op == UOps.CONST:
          reg = self.alloc_register()
          #self.mov_rr(reg, )
          #self.mov(self.buffer_registers[src[0].arg][0], src[2].arg)
          self.dealloc_register(reg)
        pass
      else: assert False, f"op {uop} not implemented" 

      i+=1

    self.xor(X86R.RAX, X86R.RAX)
    with open("/tmp/tinygradout", "wb+") as f:
      f.write(self.code)
    print(subprocess.check_output(["objdump", "-b", "binary", "-D", "-Mintel,x86-64", "-m", "i386", "/tmp/tinygradout"]).decode())
    assert False
    return i

  def render(self, name: str, uops: List[UOp]) -> str:
    instructions_parsed = self.render_recursive(uops)
    print(instructions_parsed, self.code)
    return str(self.code)

class ASMRenderer(Renderer):
  device = "ASM"
  global_max = None
  has_local = False

  def __init__(self) -> None:
    if platform.machine() == 'x86_64':
      with open("/proc/cpuinfo", "r") as f:
        cpuinfo = f.read()
        # TODO: probably not a good way to do this.
        avx = cpuinfo.find("avx ") != -1; avx2 = cpuinfo.find("avx2 ") != -1; avx512 = cpuinfo.find("avx512 ") != -1
        self.renderer = X8664ASMRenderer(avx, avx2, avx512)
    else: raise RuntimeError(f'Architecture {platform.machine()} not supported for assembly backend.')

  def render(self, name: str, uops: List[UOp]) -> str:
    return self.renderer.render(name, uops)

import subprocess

if __name__ == "__main__":
  renderer = X8664ASMRenderer()
  renderer.xor(X86R.R9, X86R.RAX)
  renderer.xor(X86R.RAX, X86R.RAX)
  renderer.sub(X86R.RCX, 1)
  renderer.sub(X86R.RDX, 1)
  renderer.sub(X86R.RAX, 1)
  renderer.sub(X86R.RBX, 1)
  renderer.sub(X86R.R8, 1)
  renderer.sub(X86R.R9, 1)
  renderer.sub(X86R.R12, 1)
  renderer.jne(-(renderer.tell()))
  print(renderer.code.hex())
  with open("/tmp/tinygradout", "wb+") as f:
    f.write(renderer.code)
  print(subprocess.check_output(["objdump", "-b", "binary", "-D", "-Mintel,x86-64", "-m", "i386", "/tmp/tinygradout"]).decode())
