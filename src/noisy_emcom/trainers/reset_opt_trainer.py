from typing import Callable

import jax

from noisy_emcom.utils import types
from noisy_emcom.utils.population_storage import PopulationStorage


class ResetOptTrainer:
    def __init__(self):
        self._speaker = "speaker"
        self._listener = "listener"

    def reset(
        self,
        population_storage: PopulationStorage,
        opt_speaker_init_fn: Callable,
        opt_listener_init_fn: Callable,
        reset_mode: types.ResetMode,
    ) -> PopulationStorage:
        agent_data = population_storage.snapshot()
        assert len(agent_data.params["speaker"]) == len(agent_data.params["listener"])
        opt_speaker_init_pmap = jax.pmap(opt_speaker_init_fn)
        opt_listener_init_pmap = jax.pmap(opt_listener_init_fn)

        for i in range(len(agent_data.params["speaker"])):
            self._reset_opt_states(
                population_storage,
                opt_speaker_init_pmap,
                agent_data.params[self._speaker][i],
                self._speaker,
                i,
                reset_mode,
            )

            self._reset_opt_states(
                population_storage,
                opt_listener_init_pmap,
                agent_data.params[self._listener][i],
                self._listener,
                i,
                reset_mode,
            )

        return population_storage

    def _reset_opt_states(
        self,
        population_storage: PopulationStorage,
        init_fn: Callable,
        agent_params: types.Params,
        agent: str,
        agent_idx: int,
        reset_mode: types.ResetMode,
    ) -> None:
        if (reset_mode == types.ResetMode.SPEAKER and agent == "listener") or (
            reset_mode == types.ResetMode.LISTENER and agent == "speaker"
        ):
            return

        new_params = init_fn(agent_params)
        population_storage.store_agent(agent, agent_idx, None, None, new_params)
