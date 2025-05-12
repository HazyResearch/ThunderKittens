from pathlib import Path

from kvm_unity.kvm import KVM_Interpreter
from kvm_unity.latency.kvm import LatencyKVM_Interpreter
from kvm_unity.latency.python_vm import (
    INSTRUCTION_TO_SOLVER as LATENCY_INSTRUCTION_TO_SOLVER,
)
from kvm_unity.latency.scheduler import LatencyScheduleBuilder
from kvm_unity.python_vm import PyVM_Interpreter
from kvm_unity.scheduler import ScheduleBuilder

BUILDER_MAP = {
    "latency": LatencyScheduleBuilder,
}

KVM_INTERPRETER_MAP = {
    "latency": LatencyKVM_Interpreter,
}

INSTRUCTION_TO_SOLVER_MAP = {
    "latency": LATENCY_INSTRUCTION_TO_SOLVER,
}


def make_schedule_builder(mode: str) -> ScheduleBuilder:
    return BUILDER_MAP[mode]()


def make_kvm_interpreter(mode: str, kvm_dir: Path) -> KVM_Interpreter:
    return KVM_INTERPRETER_MAP[mode](kvm_dir)


def make_pyvm_interpreter(mode: str) -> PyVM_Interpreter:
    return PyVM_Interpreter(INSTRUCTION_TO_SOLVER_MAP[mode])
