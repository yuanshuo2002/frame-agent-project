# FRAME Agents
from .base_agent import BaseAgent
from .generator import GeneratorAgent, TrainingGenerator
from .evaluator import EvaluatorAgent, compute_section_average
from .reflector import ReflectorAgent, FrameTrainingLoop
