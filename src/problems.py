from Agents import *
from abc import ABC, abstractmethod

# Class to define neighbor relationships between agents in a DCOP
class Neighbors():
    def __init__(self, a1:Agent, a2:Agent,dcop_id):

        if a1.id_<a2.id_:
            self.a1 = a1
            self.a2 = a2
        else:
            self.a1 = a2
            self.a2 = a1

        self.dcop_id = dcop_id
        self.rnd_cost = random.Random((((dcop_id+1)+100)+((a1.id_+1)+10)+((a2.id_+1)*1))*17)
        self.cost_table = {}
        self.create_dictionary_of_costs()  # Initialize the cost table with random costs

    # Get the cost associated with the given variables of two agents
    def get_cost(self, a1_id, a1_variable, a2_id, a2_variable):
        ap =((str(a1_id),a1_variable),(str(a2_id),a2_variable))
        ans = self.cost_table[ap]
        return ans

    # Create a dictionary of costs for all possible combinations of agent variables
    def create_dictionary_of_costs(self):
        for d_a1 in self.a1.domain:
            for d_a2 in self.a2.domain:
                first_tuple = (str(self.a1.id_),d_a1)
                second_tuple = (str(self.a2.id_),d_a2)
                ap = (first_tuple,second_tuple)
                min_cost=0
                max_cost=100
                cost = self.rnd_cost.randint(min_cost, max_cost)
                self.cost_table[ap] = cost

    # Check if a given agent is part of this neighbor relationship
    def is_agent_in_obj(self,agent_id_input):
        if agent_id_input == self.a1.id_  or agent_id_input == self.a2.id_ :
            return True
        return False

    # Get the other agent in the neighbor relationship
    def get_other_agent(self,agent_input):
        if agent_input.id_ == self.a1.id_:
            return self.a2.id_
        else:
            return self.a1.id_


# Class to define an unbounded message buffer
class UnboundedBuffer():

    def __init__(self):
        self.buffer = []

    # Insert a list of messages into the buffer
    def insert(self, list_of_msgs):
        for msg in list_of_msgs:
            self.buffer.append(msg)

    # Extract all messages from the buffer
    def extract(self):
        ans = []
        for msg in self.buffer:
            if msg is None:
                return None
            else:
                ans.append(msg)
        self.buffer = []
        return ans

# Class to manage message passing between agents in a DCOP
class Mailer():
    def __init__(self,agents):
        self.inbox = UnboundedBuffer()
        self.agents_outbox = {}
        for a in agents:
            outbox = UnboundedBuffer()
            self.agents_outbox[a.id_] = outbox
            a.inbox = outbox
            a.outbox = self.inbox

    # Place messages in the agents' inboxes
    def place_messages_in_agents_inbox(self):
        msgs_to_send = self.inbox.extract()
        if len(msgs_to_send) == 0: return True
        msgs_by_receiver_dict = self.create_msgs_by_receiver_dict(msgs_to_send)
        for receiver,msgs_list in msgs_by_receiver_dict.items():
            self.agents_outbox[receiver].insert(msgs_list)

    # Create a dictionary of messages categorized by their receivers
    def create_msgs_by_receiver_dict(self,msgs_to_send):
        msgs_by_receiver_dict = {}
        for msg in msgs_to_send:
            receiver = msg.receiver
            if receiver not in msgs_by_receiver_dict.keys():
                msgs_by_receiver_dict[receiver] = []
            msgs_by_receiver_dict[receiver].append(msg)
        return msgs_by_receiver_dict

# Abstract base class for DCOP problems
class DCOP(ABC):
    def __init__(self,id_,A,D,dcop_name,algorithm, k, p = None):
        self.dcop_id = id_
        self.A = A  # Number of agents
        self.D = D  # size of domain
        self.k = k  # Probability of edge creation between agents
        self.p=p  # Probability parameter for DSA
        self.algorithm = algorithm  # Algorithm to be used
        self.dcop_name = dcop_name
        self.agents = []
        self.create_agents()  # Initialize agents
        self.neighbors = []
        self.rnd_neighbors = random.Random((id_+5)*17)
        self.create_neighbors()
        self.connect_agents_to_neighbors()
        self.mailer = Mailer(self.agents)  # Initialize mailer
        self.global_clock = 0
        self.global_costs=[]

    # Connect each agent to its neighbors
    def connect_agents_to_neighbors(self):
        for a in self.agents:
            neighbors_of_a = self.get_all_neighbors_obj_of_agent(a)
            a.set_neighbors(neighbors_of_a)

    # Get all neighbor objects for a given agent
    def get_all_neighbors_obj_of_agent(self, agent:Agent):
        ans = []
        for n in self.neighbors:
            if n.is_agent_in_obj(agent.id_):
                ans.append(n)
        return ans

    # Calculate the global cost of the current state
    def calc_global_cost(self):
        global_cost = 0
        for n in self.neighbors:
            a1_id, a1_variable= n.a1.id_, n.a1.variable
            a2_id, a2_variable= n.a2.id_, n.a2.variable
            global_cost += n.get_cost(a1_id, a1_variable, a2_id, a2_variable)
        return global_cost

    # Execute the DCOP algorithm
    def execute(self):
        initial_global_cost =self.calc_global_cost()
        self.global_costs.append(initial_global_cost)

        for a in self.agents:
            a.initialize()

        for i in range(incomplete_iterations):
            self.global_clock = self.global_clock + 1
            is_empty = self.mailer.place_messages_in_agents_inbox()
            if is_empty:
                print("DCOP:",str(self.dcop_id),"global clock:",str(self.global_clock), "is over because there are no messages in system ")
                break
            self.agents_perform_iteration(self.global_clock)
            current_global_cost=self.calc_global_cost()
            self.global_costs.append(current_global_cost)

        return self.global_costs

    # Create neighbors based on the probability k
    def create_neighbors(self):
        for i in range(self.A):
            a1 = self.agents[i]
            for j in range(i+1,self.A):
                a2 = self.agents[j]
                rnd_number = self.rnd_neighbors.random()
                if rnd_number<self.k:
                    self.neighbors.append(Neighbors(a1, a2, self.dcop_id))

    # Perform an iteration for all agents
    def agents_perform_iteration(self,global_clock):
        for a in self.agents:
            a.execute_iteration(global_clock)

    # Abstract method to create agents, to be implemented by subclasses
    @abstractmethod
    def create_agents(self):
        pass


# Class for DCOP using the DSA algorithm
class DCOP_DSA(DCOP):

    def __init__(self, id_,A,D,dcop_name,algorithm, k, p):
        DCOP.__init__(self,id_,A,D,dcop_name,algorithm, k, p)

    # Create DSA agents
    def create_agents(self):
        for i in range(self.A):
            self.agents.append(DSA_Agent(i + 1, self.D, self.p))


# Class for DCOP using the MGM algorithm
class DCOP_MGM(DCOP):

    def __init__(self, id_,A,D,dcop_name,algorithm, k):
        DCOP.__init__(self,id_,A,D,dcop_name,algorithm, k)

    # Create MGM agents
    def create_agents(self):
        for i in range(self.A):
            self.agents.append(MGM_Agent(i + 1, self.D))