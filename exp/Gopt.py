import argparse
import math
from gurobipy import Model, GRB
import scipy.optimize as spo
import os
import sys
import json
import time
from decimal import Decimal, getcontext


# Queueing time distribution function
def waitingTimeDistr(t, lm, service_rate): 
    mu = service_rate
    if lm >= mu:
        return 0.0
    if lm <= 0:
        return 1.0
    
    getcontext().prec = 80
    s = Decimal(0)
    dlm = Decimal(lm)
    dmu = Decimal(mu)
    dt = Decimal(t)
    for i in range(int(t * mu) + 1):
        di = Decimal(i)
        t1 = di / dmu - dt
        f1 = Decimal.exp(-dlm * t1)
        f2 = t1 ** di
        f3 = dlm ** di
        f4 = math.factorial(i)
        s = s + f1 * f2 * f3 / f4    
    return float((Decimal(1) - dlm / dmu) * s)



def initialize_model(sites, pricing, opti_pref, applications, capacities, theta):
    # Initialize the model
    model = Model("Cerberus_Gopt")

    # Set Gurobi parameters
    model.setParam(GRB.Param.TimeLimit, 10000)  # Limit the solver to 10000 seconds
    model.setParam(GRB.Param.MIPFocus, 1)    # Focus on finding feasible solutions faster
    model.setParam(GRB.Param.Cuts, 3)        # Use aggressive cutting planes
    model.setParam(GRB.Param.Presolve, 1)    # Enable strong presolve to reduce problem size
    model.setParam(GRB.Param.Heuristics, 0.9) # Increase heuristic search

    # Decision variables
    x = {} # Assignment of users to sites
    y = {} # Open sites
    y_site = model.addVars(sites, vtype=GRB.BINARY, name="y_site")
    # y = model.addVars(sites, vtype=GRB.BINARY, name=f"y")  # Moving it to global constrant to make the min_site available
    u = {} # Number of users per site
    rtt = {}
    arrivalRateFromUserToSite = {}
    instances_per_app_site = {}
    instance_arrival_rate = {}
    lambdaLimits = {}

    # Define M as a large constant
    M = 1e6

    # Constraints and lambda for each app
    for app, app_data in applications.items():
        users = app_data["users"]
        request_rates = app_data["request_rates"]
        latency_site_user = {(i, j): app_data["latency_site_user"][f"('{i}', '{j}')"] for i in sites for j in users}
        service_rate = app_data["service_rate"]
        slo = app_data["slo"]

        x[app] = model.addVars(sites, users, vtype=GRB.BINARY, name=f"x_{app}")  # Assignment of users to sites
        y[app] = model.addVars(sites, vtype=GRB.BINARY, name=f"y_{app}")  # Site is open
        u[app] = model.addVars(sites, vtype=GRB.INTEGER, name=f"u_{app}")  # Number of users per site

        arrivalRateFromUserToSite[app] = model.addVars(sites, users, vtype=GRB.CONTINUOUS, name=f"arrivalRateFromUserToSite_{app}")  # arrival rate from user to site
        rtt[app] = model.addVars(sites, users, lb=-M, vtype=GRB.CONTINUOUS, name=f"rtt_{app}")  # Round trip time

        # Arrival rate at instance
        instance_arrival_rate[app] = model.addVars(sites, vtype=GRB.CONTINUOUS, name=f"utilization_{app}")
        # instance_per_app
        instances_per_app_site[app] = model.addVars(sites, vtype=GRB.INTEGER, name=f"instances_{app}")

        # Constraints

        # Each user is assigned to exactly one site
        model.addConstrs((x[app].sum('*', j) == 1 for j in users), name=f"assignToOne_{app}")

        # Users can only be linked to the open sites
        model.addConstrs((x[app][i, j] <= y[app][i] for i in sites for j in users), name=f"linkToOpen_{app}")

        # Enforce instance arrival rate
        model.addConstrs((arrivalRateFromUserToSite[app][i, j] == x[app][i, j] * request_rates[j] for i in sites for j in users), name=f"user_site_arrival_rate_{app}")

        # Arrival rate at the instances
        model.addConstrs((instance_arrival_rate[app][i] * instances_per_app_site[app][i] == arrivalRateFromUserToSite[app].sum(i, '*') for i in sites), name=f"instance_arrival_rate_{app}")
        
        # Number of users per site
        model.addConstrs((u[app][i] == x[app].sum(i, '*') for i in sites), name=f"numUsers_{app}")

        # If a site is open, it should have at least one instance deployed, no matter which application
        model.addConstrs((instances_per_app_site[app][i] >= y[app][i] for i in sites), name=f"atLeastOneInstanceIfOpen_{app}")

        # Lambda limits for each user and site
        lambdaLimits[app] = {i: {j: spo.brentq(lambda lm: waitingTimeDistr(slo - latency_site_user[i, j] - 1 / service_rate, lm, service_rate) - theta, 0.0, service_rate) for j in users} for i in sites}

        # Enforce maximum lambda
        for j in users:
            for i in sites:
                model.addConstr(x[app][i, j] * instance_arrival_rate[app][i] <= lambdaLimits[app][i][j], name=f"lambdaMax_{app}_{i}_{j}")

    #########################################
    # Global constraints for all the apps
    #########################################

    # Capacity constraints: number of instances of all application on one site shouldn't be bigger than the number of slots
    model.addConstrs((sum(instances_per_app_site[app][i] * applications[app]["slots"] for app in applications) <= capacities[i] for i in sites), name="capacity")

    # Link y_site with y[app][i], new!!
    for i in sites:
        for app in applications:
            model.addConstr(y[app][i] <= y_site[i], name=f"link_{app}_{i}")

    # Objective function
    if opti_pref == 1:
        # Minimize the number of open sites for all the applications
        # model.setObjective(sum(y[app][i] for app in applications for i in sites), GRB.MINIMIZE)
        model.setObjective(sum(y_site[i] for i in sites), GRB.MINIMIZE)
    elif opti_pref == 2:
        # model.setObjective(sum(instances_per_app_site[app].sum() for app in applications), GRB.MINIMIZE)
        model.setObjective(sum(instances_per_app_site[app][i] * applications[app]["slots"] for app in applications for i in sites), GRB.MINIMIZE)
    elif opti_pref == 3:
        # Minimize the financial cost
        model.setObjective(sum(instances_per_app_site[app][i] * applications[app]["slots"] * pricing[i] for app in applications for i in sites), GRB.MINIMIZE)
    else:
        print("No optimization preference selected.")
        sys.exit(1)

    return model, x, y, u, instance_arrival_rate, instances_per_app_site


# Optimize the model
def run_optimization(model):
    model.Params.DisplayInterval = 500
    model.optimize()


# Generate the placement plan JSON file
def placement_plan_json(sites, applications, x, y, instances_per_app_site, exp_round_index):
    placement_plan = {
        "sites": []
    }

    for i in sites:
        site_info = {
            "site": i,
            "applications": []
        }

        for app, app_data in applications.items():
            if y[app][i].X > 0.5:
                app_info = {
                    "app": app,
                    "instances": math.ceil(instances_per_app_site[app][i].X),
                    "users": []
                }

                for j in app_data["users"]:
                    if x[app][i, j].X > 0.5:
                        user_info = {
                            "user": j,
                            "request_rate": app_data["request_rates"][j]
                        }
                        app_info["users"].append(user_info)

                site_info["applications"].append(app_info)

        placement_plan["sites"].append(site_info)

    # Write the placement plan to a JSON file
    with open(f"placement_plan_r{exp_round_index}.json", "w") as json_file:
        json.dump(placement_plan, json_file, indent=4)

    print("Placement plan has been saved to placement_plan.json")




def print_results(sites, applications, x, y, u, instance_arrival_rate, instances_per_app_site, theta, exp_round_index, slt_introduced_apps=None, priority_levels=None, rounds=None, gap_slo=None):
    print("====================================================================================================")
    if slt_introduced_apps and priority_levels is not None:
        print(f"**** SLT Introduced for priority levels {', '.join(map(str, priority_levels))}: {', '.join(slt_introduced_apps)} ****")
    if rounds is not None:
        print(f"**** Optimization completed in {rounds} rounds ****")

    for i in sites:
        # If a site is open for all the apps, site is then closed
        if all(y[app][i].X < 0.5 for app in applications):
            print("----------------------------------------------------------------------------------------------------")
            print(f"Site {i} is closed.")
        for app, app_data in applications.items():

            if y[app][i].X > 0.5:
                service_rate = app_data["service_rate"]
                slo = app_data["slo"]
                users = app_data["users"]
                latency_site_user = {(i, j): app_data["latency_site_user"][f"('{i}', '{j}')"] for j in users}
                rho = instance_arrival_rate[app][i].X / service_rate

                print("----------------------------------------------------------------------------------------------------")
                print(f"Site {i} is open for app: {app} with {math.ceil(instances_per_app_site[app][i].X)} instances, {u[app][i].X} users assigned.")
                print(f"    Arrival rate at the instances for *{app}*: {instance_arrival_rate[app][i].X:.5f}, Utilization of instances: {rho:.5f}")
                if gap_slo and app in gap_slo:
                    print(f"*** SLT used for {app}: {applications[app]['slo']}, compromising {gap_slo[app]} ms ***")
                else:
                    print(f"*** SLO used for {app}: {applications[app]['slo']} ***")
                
                for j in users:
                    if x[app][i, j].X > 0.5:                    
                        rtt_value = latency_site_user[i, j] + 1 / service_rate + rho / (2 * service_rate * (1 - rho))
                        perc = spo.brentq(lambda t: waitingTimeDistr(t - latency_site_user[i, j] - 1 / service_rate, instance_arrival_rate[app][i].X, service_rate) - theta, 0, slo + 1)
                        print(f"  User {j} with average RTT {rtt_value} and {theta * 100}% percentile RTT {perc}.")
                        if gap_slo and app in gap_slo:
                            print(f"    Probability that the requests of this user will stay below the SLO-RTT: {waitingTimeDistr(slo - latency_site_user[i, j] - 1 / service_rate, instance_arrival_rate[app][i].X, service_rate)}")
                        else:
                            print(f"    Probability that the requests of this user will stay below the SLT-RTT: {waitingTimeDistr(slo - latency_site_user[i, j] - 1 / service_rate, instance_arrival_rate[app][i].X, service_rate)}")
    print("====================================================================================================")

    # Generate the placement plan JSON file
    placement_plan_json(sites, applications, x, y, instances_per_app_site, exp_round_index)




def main ():

    parser = argparse.ArgumentParser(description='Run the Multi-LASSY optimization.')
    parser.add_argument('--round', type=int, required=True, help='The round index for the JSON file.')
    args = parser.parse_args()

    exp_round_index = args.round

    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Path to the JSON file
    json_file = os.path.join(script_dir, 'input_high/a6.json')

    # Load the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    sites = data["sites"]
    capacities = data["capacities"]
    pricing = data["pricing"]
    opti_pref = data["opti_pref"]
    applications = data["applications"]
    theta = 0.99 # percentile


    start_time = time.time()
    model, x, y, u, instance_arrival_rate, instances_per_app_site = initialize_model(sites, pricing, opti_pref, applications, capacities, theta)
    

    run_optimization(model)
    


    if model.status == GRB.OPTIMAL:
        print_results(sites, applications, x, y, u, instance_arrival_rate, instances_per_app_site, theta, exp_round_index, rounds = 1)
        print(f"Objective value is: {model.ObjVal}")
    elif model.status == GRB.TIME_LIMIT:
        print("Time limit reached. No optimal solution found. The feasible solution is:")
        print_results(sites, applications, x, y, u, instance_arrival_rate, instances_per_app_site, theta, exp_round_index)
        print(f"Objective value is: {model.ObjVal}")
    elif model.status == GRB.INFEASIBLE:
        print("No optimal solution found.")
    

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
if __name__ == "__main__":
    main()