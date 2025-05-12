import torch  # type: ignore
import numpy as np # type: ignore
import random
from collections import deque
from snake_game import SnakeGameAI,Direction,Point
from model import QTrainer,Linear_QNet
from plotter import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000 # max batch size for training through long term memory
LR=0.001 # learning rate

# STEPS:
# 1 get current state 
# 2 get action according to current state
# 3 reward ,game_over,score=game.play_step(action)
# 4 get new state 
# 5 store everything in memory 
# 6 train the model acc
# Gamma: 0.9 to 0.99
# Learning Rate: 0.1 to 0.01

class Agent:
    def __init__(self):
        self.no_game=0
        self.epsilon=0 # to control randomness
        self.gamma=0.9 # Discount rate (set accordingly but must be smaller than 1)
        self.memory=deque(maxlen=MAX_MEMORY)
        self.model=Linear_QNet(11,256,3)  #(n_state,hidden_layer,n_action)
        self.trainer=QTrainer(self.model,lr=LR,gamma=self.gamma)

    def get_state(self,game):
        head=game.snake[0] #get the head of snake form the list
        #points to check for potential collisions (BLOCK_SIZE=20 therefore we could predict a collision just a block before)
        point_l=Point(head.x-20,head.y)
        point_r=Point(head.x+20,head.y)
        point_u=Point(head.x,head.y-20)
        point_d=Point(head.x,head.y+20)

        #current possible game directions
        dir_l=game.direction==Direction.LEFT
        dir_r=game.direction==Direction.RIGHT
        dir_u=game.direction==Direction.UP
        dir_d=game.direction==Direction.DOWN

        state=[
            #DANGER straight
            #if we are going for example in direction right and the point right gives a collision then we have DANGER straight away
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            #DANGER right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            #DANGER left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            #MOVE direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            #Food Location
            game.food.x < game.head.x, # food is left from us
            game.food.x > game.head.x, # food is right from us
            game.food.y < game.head.y, # food is up from us
            game.food.y > game.head.y, # food is down from us
        ]
        return np.array(state,dtype=int)
        
    def remember(self,state,action,reward,next_state,done): #game_over=done
        self.memory.append((state,action,reward,next_state,done)) #popleft if max_memory is reached
        
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE: 
            mini_sample=random.sample(self.memory,BATCH_SIZE) # get a list of 1000 tuples (state,action,reward,next_state,done) to train on
        else:
            mini_sample=self.memory

        states,actions,rewards,next_states,dones = zip(*mini_sample)
        self.trainer.train_step(states,actions,rewards,next_states,dones)
        
    def train_short_memory(self,state,action,reward,next_state,done):
        self.trainer.train_step(state,action,reward,next_state,done)
    def get_action(self,state):
        # intial we explore the enviroment through random moves
        #as the model learns and gets experience we will then exploit the agent
        # as the number of games increases the epsilon will gradually decrease and limit the random moves selection
        self.epsilon = 80- self.no_game
        final_move=[0,0,0]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            #moves form model
            state0=torch.tensor(state, dtype=torch.float)
            prediction= self.model(state0)  # executes the forward function in model.py
            move=torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move

def train():
    plot_scores=[]
    plot_mean_scores=[]
    total_score=0
    record=0
    agent=Agent()
    game=SnakeGameAI()
    #training loop
    while True:
        #get old state
        current_state=agent.get_state(game)
        #get action
        final_move=agent.get_action(current_state)
        #perform action and get new state and get state evaluation acc
        reward,done,score = game.play_step(final_move)
        new_state=agent.get_state(game)

        #train short memory for step
        agent.train_short_memory(current_state,final_move,reward,new_state,done)

        #remember
        agent.remember(current_state,final_move,reward,new_state,done)

        if done:
            #train long memory
            game.reset()
            agent.no_game+=1
            agent.train_long_memory()

            if score>record:
                record=score
                agent.model.save_model()


            print('GAME',agent.no_game, 'SCORE',score, 'RECORD',record)
            
            #plotting
            plot_scores.append(score)
            total_score+=score
            mean_score= total_score/agent.no_game
            plot_mean_scores.append(mean_score)
            plot(plot_scores,plot_mean_scores)
            
if __name__ == "__main__":
    train()