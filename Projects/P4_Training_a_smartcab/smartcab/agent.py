import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from numpy.random import choice
from os import remove
import pandas as pd

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        valid_waypoints = ['forward','right','left']
        valid_lights = ['red','green']
        valid_actions = ['forward','right','left', None]

        self.Q_hat={}
        states = [(next_waypoint,light,oncoming,right,left) for next_waypoint in valid_waypoints for light in valid_lights for oncoming in valid_actions for right in valid_actions for left in valid_actions]
        states.append('Destination')
        for state in states:
            self.Q_hat[state] = {}
            for action in valid_actions:
                self.Q_hat[state][action] = random.random()

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        if self.next_waypoint == None:
            self.state = 'Destination'
        else:
            self.state = (self.next_waypoint, inputs['light'],inputs['oncoming'],inputs['right'],inputs['left'])

        # TODO: Select action according to your policy

        valid_actions = ['forward','right','left', None]
        epsilon = 0.01 #0.01

        best_action = max(self.Q_hat[self.state],key=self.Q_hat[self.state].get)
        random_action = choice(valid_actions)
        action = choice([best_action, random_action],p=[1-epsilon,epsilon])

        # Execute action and get reward

        reward = self.env.act(self, action)

        # Learn policy based on state, action, reward

        new_next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        new_inputs = self.env.sense(self)
        new_state = (new_next_waypoint, new_inputs['light'],new_inputs['oncoming'],new_inputs['right'],new_inputs['left'])
        alpha = 0.7 #opt 0.7
        gamma = 0.1 #opt 0.1
        max_Qhat_ahat = max(self.Q_hat[new_state].values())
        self.Q_hat[self.state][action] = (1-alpha)*self.Q_hat[self.state][action]+alpha*(reward+gamma*max_Qhat_ahat)

        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

def run():
    """Run the agent for a finite number of trials."""
    remove('logfile')
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    log = pd.read_csv('logfile',names=['status','reward'])
    success_rate = 1.0*len(log.loc[log.status == 'Reached'])/len(log)
    mean_reward = log.loc[log.status == 'Reached']['reward'].mean()
    print "Expected reward = "+str(success_rate*mean_reward)


if __name__ == '__main__':
    run()
