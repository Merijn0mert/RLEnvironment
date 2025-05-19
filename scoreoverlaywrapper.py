import cv2
import numpy as np
from gymnasium import Wrapper


class ScoreOverlayWrapper(Wrapper):
    # This class adds the score overlay for the evaluatevideo function in dqn_agent.py
    def __init__(self, env):
        super().__init__(env)
        self.score = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.score = info.get("score", self.score)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.score = 0
        return self.env.reset(**kwargs)

    def render(self):
        frame = self.env.render()
        # Add score text to the frame
        if frame is not None:
            frame = frame.copy()
            cv2.putText(
                frame,
                f"Score: {self.score}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return frame
