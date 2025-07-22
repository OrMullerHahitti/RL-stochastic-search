from enum import Enum

# Define the number of repetitions and incomplete iterations
repetitions = 30
incomplete_iterations = 100

# Define a class to represent messages exchanged between agents

class Msg():

    def __init__(self, sender, receiver, information):
        self.sender = sender  # The agent sending the message
        self.receiver = receiver  # The agent receiving the message
        self.information = information  # The information contained in the message

# Define an enumeration for the algorithms used in DCOPs
class Algorithm(Enum):
    DSA = 1  # DSA algorithm
    MGM = 2  # MGM algorithm
    DSA_RL = 3  # DSA with REINFORCE learning

