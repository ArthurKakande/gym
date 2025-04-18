from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.space import MultiGrid
import random

# Utility function to calculate Manhattan distance
def distance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

class Customer(Agent):
    def __init__(self, unique_id, model, origin, destination):
        super().__init__(unique_id, model)
        self.origin = origin
        self.destination = destination
        self.in_vehicle = False
        self.current_vehicle = None

    def step(self):
        if not self.in_vehicle:
            self.decide_transport()
        else:
            # Move with the vehicle to the destination
            if self.current_vehicle.pos != self.destination:
                self.current_vehicle.move_toward(self.destination)
            else:
                print(f"{self.current_vehicle.unique_id} dropped off the customer at {self.destination}.")
                self.in_vehicle = False
                self.model.running = False  # End simulation

    def decide_transport(self):
        chosen_transport = random.choice(["taxi", "bus"])
        if chosen_transport == "taxi":
            accepting_taxis = [
                taxi for taxi in self.model.schedule.agents
                if isinstance(taxi, Taxi) and taxi.evaluate_request(self.origin)
            ]
            if accepting_taxis:
                nearest_taxi = min(accepting_taxis, key=lambda taxi: distance(taxi.pos, self.origin))
                if nearest_taxi.pos != self.origin:
                    nearest_taxi.move_toward(self.origin)
                else:
                    print(f"{nearest_taxi.unique_id} picked up the customer.")
                    self.in_vehicle = True
                    self.current_vehicle = nearest_taxi
            else:
                print("No taxi accepted the request. Customer waits for the bus.")
        if not self.in_vehicle:
            nearest_bus = min(
                [agent for agent in self.model.schedule.agents if isinstance(agent, Bus)],
                key=lambda bus: distance(bus.pos, self.origin)
            )
            if nearest_bus.pos != self.origin:
                nearest_bus.move_toward(self.origin)
            else:
                print(f"{nearest_bus.unique_id} picked up the customer.")
                self.in_vehicle = True
                self.current_vehicle = nearest_bus

class Vehicle(Agent):
    def __init__(self, unique_id, model, position):
        super().__init__(unique_id, model)
        super().__init__(unique_id, model)
        self.pos = position

    def move_toward(self, target):
        x, y = self.pos
        tx, ty = target
        if x < tx:
            x += 1
        elif x > tx:
            x -= 1
        elif y < ty:
            y += 1
        elif y > ty:
            y -= 1
        self.model.grid.move_agent(self, (x, y))

class Taxi(Vehicle):
    def evaluate_request(self, customer_origin):
        buses = [agent for agent in self.model.schedule.agents if isinstance(agent, Bus)]
        nearest_bus = min(buses, key=lambda bus: distance(bus.pos, customer_origin))
        bus_time = distance(nearest_bus.pos, customer_origin)
        taxi_time = distance(self.pos, customer_origin)
        return taxi_time < bus_time

class Bus(Vehicle):
    pass

class TransportModel(Model):
    def __init__(self, width, height):
        super().__init__()  # Initialize the base Model class
        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = SimultaneousActivation(self)
        self.running = True  # Ensure the simulation can run

        # Add taxis
        taxi1 = Taxi("Taxi1", self, (random.randint(0, width - 1), random.randint(0, height - 1)))
        taxi2 = Taxi("Taxi2", self, (random.randint(0, width - 1), random.randint(0, height - 1)))
        self.grid.place_agent(taxi1, taxi1.pos)
        self.grid.place_agent(taxi2, taxi2.pos)
        self.schedule.add(taxi1)
        self.schedule.add(taxi2)

        # Add a bus
        bus = Bus("Bus1", self, (random.randint(0, width - 1), random.randint(0, height - 1)))
        self.grid.place_agent(bus, bus.pos)
        self.schedule.add(bus)

        # Add a customer
        customer_origin = (random.randint(0, width - 1), random.randint(0, height - 1))
        customer_destination = (random.randint(0, width - 1), random.randint(0, height - 1))
        customer = Customer("Customer", self, customer_origin, customer_destination)
        self.grid.place_agent(customer, customer_origin)
        self.schedule.add(customer)

    def step(self):
        self.schedule.step()

# Run the simulation
model = TransportModel(10, 10)
while model.running:
    model.step()