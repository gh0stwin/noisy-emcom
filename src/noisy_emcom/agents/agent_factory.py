"""Agent Factory."""

from noisy_emcom.agents.lewis_game_agent import LewisGameAgent
from noisy_emcom.utils import types


def agent_factory(agent_type: str, agent_kwargs: types.Config) -> LewisGameAgent:
    """Lewis game agent factory."""
    if agent_type == types.AgentType.LEWIS_GAME:
        agent = LewisGameAgent(**agent_kwargs)
    else:
        raise ValueError(f"Unknown agent type: '{agent_type}'")

    return agent
