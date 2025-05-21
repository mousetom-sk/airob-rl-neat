import numpy as np
import pygame

from environment import Environment


actions = {
    pygame.K_KP_1: np.array([-1, -1]),
    pygame.K_KP_2: np.array([0, -1]),
    pygame.K_KP_3: np.array([1, -1]),
    pygame.K_KP_4: np.array([-1, 0]),
    pygame.K_KP_5: np.array([0, 0]),
    pygame.K_KP_6: np.array([1, 0]),
    pygame.K_KP_7: np.array([-1, 1]),
    pygame.K_KP_8: np.array([0, 1]),
    pygame.K_KP_9: np.array([1, 1])
}


if __name__ == "__main__":
    env = Environment(1000)
    done = True

    while True:
        if done:
            env.reset()
            done = False

        pygame.event.wait(200)

        act = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        
        keys = pygame.key.get_pressed()
        for k, v in actions.items():
            if keys[k]:
                act = np.copy(v)
                break
        
        if act is not None:
            done = env.step(act)[-2]
