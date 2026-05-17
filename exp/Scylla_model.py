# Scylla integrated with LPS
# With y_site binary variable for site open/close decision integrated with y[app][i]
import argparse
import math
import random
from gurobipy import Model, GRB
import scipy.optimize as spo
import os
import sys
import time
import itertools
import json
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
    model = Model("Multi-LASSY")
    model.setParam('Threads', 4)

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
    model.Params.OutputFlag = 0  # Suppress Gurobi output
    # model.Params.DisplayInterval = 300
    model.optimize()


# Update the universal placement plan
def update_universal_placement_plan(universal_placement_plan, sites, applications, x, y, instances_per_app_site):
    for i in sites:
        for app in applications:
            if y[app][i].X > 0.5:
                app_info = {
                    "app": app,
                    "instances": math.ceil(instances_per_app_site[app][i].X),
                    "users": []
                }
                for j in applications[app]["users"]:
                    if x[app][i, j].X > 0.5:
                        user_info = {
                            "user": j,
                            "request_rate": applications[app]["request_rates"][j]
                        }
                        app_info["users"].append(user_info)
                universal_placement_plan["sites"][i]["applications"].append(app_info)


# Convert the universal placement plan to the required format
def convert_universal_placement_plan(universal_placement_plan):
    final_placement_plan = {
        "sites": []
    }

    for site, site_data in universal_placement_plan["sites"].items():
        site_info = {
            "site": site,
            "applications": site_data["applications"]
        }
        final_placement_plan["sites"].append(site_info)

    return final_placement_plan

# Generate the placement plan JSON file
def placement_plan_json(universal_placement_plan, exp_round_index):
    final_placement_plan = convert_universal_placement_plan(universal_placement_plan)

    # Write the final placement plan to a JSON file
    with open(f"placement_plan_r{exp_round_index}.json", "w") as json_file:
        json.dump(final_placement_plan, json_file, indent=4)

    print(f"Placement plan has been saved to placement_plan_r{exp_round_index}.json")



def print_results(sites, applications, x, y, u, instance_arrival_rate, instances_per_app_site, theta):
    print("====================================================================================================")
    for i in sites:
        
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
                print(f"*** SLO used for {app}: {applications[app]['slo']} ***")
                
                for j in users:
                    if x[app][i, j].X > 0.5:                    
                        rtt_value = latency_site_user[i, j] + 1 / service_rate + rho / (2 * service_rate * (1 - rho))
                        perc = spo.brentq(lambda t: waitingTimeDistr(t - latency_site_user[i, j] - 1 / service_rate, instance_arrival_rate[app][i].X, service_rate) - theta, 0, slo + 1)
                        print(f"  User {j} with average RTT {rtt_value} and {theta * 100}% percentile RTT {perc}.")
                        print(f"    Probability that the requests of this user will stay below the SLT-RTT: {waitingTimeDistr(slo - latency_site_user[i, j] - 1 / service_rate, instance_arrival_rate[app][i].X, service_rate)}")
    print("====================================================================================================")



def print_results_slt(sites, applications, x, y, u, instance_arrival_rate, instances_per_app_site, theta, slt_introduced_apps=None, priority_levels=None, rounds=None, gap_slo=None):
    print("====================================================================================================")
    if slt_introduced_apps and priority_levels is not None:
        print(f"**** App {slt_introduced_apps} degraded in priority levels {priority_levels} ****")
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
                
                for j in users:
                    if x[app][i, j].X > 0.5:                    
                        rtt_value = latency_site_user[i, j] + 1 / service_rate + rho / (2 * service_rate * (1 - rho))
                        perc = spo.brentq(lambda t: waitingTimeDistr(t - latency_site_user[i, j] - 1 / service_rate, instance_arrival_rate[app][i].X, service_rate) - theta, 0, slo + 1)
                        print(f"  User {j} with average RTT {rtt_value} and {theta * 100}% percentile RTT {perc}.")
                        if gap_slo and app in gap_slo:
                            print(f"    Probability that the requests of this user will stay below the SLO-RTT: {waitingTimeDistr(slo - latency_site_user[i, j] - 1 / service_rate, instance_arrival_rate[app][i].X, service_rate)}")
    print("====================================================================================================")


def find_best_app_permutation(sites, capacities, pricing, opti_pref, applications, theta, pi=None):
    
    # Set up the initial minimum objective value based on optimization preference
    if opti_pref == 1:
        min_objval_permutations = len(sites)
    elif opti_pref == 2:
        min_objval_permutations = sum(capacities.values())
    else: 
        min_objval_permutations = sum(capacities[i] * pricing[i] for i in sites)

    apps_permutations = list(itertools.permutations(applications.keys()))

    if pi and len(apps_permutations) > pi:

        # Having the first half of the permutations
        # apps_permutations = apps_permutations[:pi]

        # Having the last half of the permutations
        # apps_permutations = apps_permutations[-pi:]

        # 1/4
        # apps_permutations = apps_permutations[:pi]
        # apps_permutations = apps_permutations[pi:2*pi]
        # apps_permutations = apps_permutations[2*pi:3*pi]
        # apps_permutations = apps_permutations[3*pi:]

        # 1/8
        apps_permutations = apps_permutations[:pi]
        # apps_permutations = apps_permutations[pi:2*pi]
        # apps_permutations = apps_permutations[2*pi:3*pi]
        # apps_permutations = apps_permutations[3*pi:4*pi]
        # apps_permutations = apps_permutations[4*pi:5*pi]
        # apps_permutations = apps_permutations[5*pi:6*pi]
        # apps_permutations = apps_permutations[6*pi:7*pi]
        # apps_permutations = apps_permutations[7*pi:]

        # apps_permutations = random.sample(apps_permutations, pi)
        print(f"Permutations are: {apps_permutations}")

    ordering_apps_final = None

    start_time = time.time()

    for apps_permutation in apps_permutations:
        remaining_capacities = capacities.copy()
        objval_each_permutation = 0

        for app in apps_permutation:
            single_app = {app: applications[app]}
            model, _, _, _, _, instances_per_app_site = initialize_model(
                sites, pricing, opti_pref, single_app, remaining_capacities, theta
            )
            run_optimization(model)

            if model.status == GRB.OPTIMAL:
                app_objval = model.objVal
                # print(f"Permutation {apps_permutation}, App {app} min cost: {app_objval}")
                objval_each_permutation += app_objval
                for i in sites:
                    remaining_capacities[i] -= instances_per_app_site[app][i].X * applications[app]["slots"]
            else:
                objval_each_permutation = float("inf")
                break
        # Have to be <= or it would miss the optimal solution
        if objval_each_permutation != float("inf"):
            print(f"Total objective value for permutation {apps_permutation}: {objval_each_permutation}")
        if objval_each_permutation <= min_objval_permutations:
            min_objval_permutations = objval_each_permutation
            ordering_apps_final = apps_permutation

    elapsed_time = time.time() - start_time

    return ordering_apps_final, elapsed_time

def iterative_slt(applications, sites, capacities, pricing, opti_pref, theta):
    # Group applications by priority levels
    priority_groups = {}
    initial_slo = {}

    for app, app_data in applications.items():
        priority = app_data["priority_group"]
        priority_groups.setdefault(priority, []).append(app)
        initial_slo[app] = app_data["slo"]

    # Sort priorities in descending order (e.g., [3, 2, 1]) and exclude highest (1)
    sorted_priorities = sorted(priority_groups.keys(), reverse=True)
    degradable_priorities = sorted_priorities[:-1]

    rounds = 1
    fully_degraded_apps = []

    # Iterate from lowest to higher degradable priorities
    for k in range(1, len(degradable_priorities) + 1):
        slt_introduced_apps = []
        priority_levels = degradable_priorities[:k]

        for priority in priority_levels:
            slt_introduced_apps.extend(priority_groups[priority])

        # Increase SLOs for all apps in the current priority levels simultaneously
        for percentage in range(120, 201, 20):  # up to 200% of original SLO
            rounds += 1
            gap_slo = {}

            for priority in priority_levels:
                for app in priority_groups[priority]:
                    original_slo = initial_slo[app]
                    slt = applications[app]["slt"]
                    new_slo = original_slo * (percentage / 100.0)

                    # Only update if not already degraded to SLT
                    if app not in fully_degraded_apps:
                        applications[app]["slo"] = new_slo
                        gap_slo[app] = new_slo - initial_slo[app]
                    else:
                        applications[app]["slo"] = applications[app]["slt"]  # Keep the already degraded SLO
                        gap_slo[app] = slt - initial_slo[app]

            # Run optimization under cumulative degradations
            ordering_apps_final_slt, _ = find_best_app_permutation(
                sites, capacities, pricing, opti_pref, applications, theta, pi=None
            )

            if ordering_apps_final_slt is None:
                print(f"** No optimal solution found after degrading app {slt_introduced_apps} "
                      f"priority levels {', '.join(map(str, priority_levels))} at {percentage}% of initial SLO. **")
                if percentage == 200:
                    fully_degraded_apps.extend(slt_introduced_apps)
                continue
            else:
                print(f"** Optimal solution found after introducing SLT for apps {slt_introduced_apps} "
                      f"priority levels {', '.join(map(str, priority_levels))} at {percentage}% of initial SLO. **")

                return ordering_apps_final_slt, slt_introduced_apps, priority_levels, rounds, gap_slo, fully_degraded_apps

    return None, None, None, rounds, None, fully_degraded_apps


def main():
    parser = argparse.ArgumentParser(description='Run the Multi-LASSY optimization.')
    parser.add_argument('--round', type=int, required=True, help='The round index for the JSON file.')
    args = parser.parse_args()
    exp_round_index = args.round

    script_dir = os.path.dirname(os.path.realpath(__file__))
    json_file = os.path.join(script_dir, 'input_low/heu_5a.json')

    with open(json_file, 'r') as f:
        data = json.load(f)

    sites = data["sites"]
    capacities = data["capacities"]
    pricing = data["pricing"]
    opti_pref = data["opti_pref"]
    applications = data["applications"]
    theta = 0.99  # percentile

    # Sampling permutations
    num_apps = len(applications)
    # pi = int(math.factorial(num_apps))
    # pi = int(math.factorial(num_apps) / 2)
    # pi = int(math.factorial(num_apps) / 4)
    pi = int(math.factorial(num_apps) / 8)


    starting_time = time.time()
# ---------- Initial optimization without SLT ----------

    degraded = False 
    universal_placement_plan = {
        "sites": {site: {"site": site, "applications": []} for site in sites}
    }

    print("==== Evaluating permutations with initial SLOs ====")
    ordering_apps_final, time_permutations = find_best_app_permutation(
        sites, capacities, pricing, opti_pref, applications, theta, pi
    )
    
    if ordering_apps_final is not None:
        print("==== Feasible placement found with initial SLOs ====")
    else:
        print("==== No feasible solution found with initial SLOs ====")
        print("==== Invoking graceful degradation with SLT introduction ====")
        ordering_apps_final, slt_apps, priority_levels, rounds, gap_slo, fully_degraded_apps = iterative_slt(
            applications, sites, capacities, pricing, opti_pref, theta
        )
        if ordering_apps_final is not None:
            degraded = True
        else:
            print("==== No feasible solution found after full degradation of all applications. Exiting. ====")
            sys.exit(1)
    ending_time = time.time()

        # ---------- Common post-optimization evaluation ----------
    if ordering_apps_final is not None:
        objval_min = 0
        remaining_capacities = capacities.copy()

        for app in ordering_apps_final:
            single_app_dict = {app: applications[app]}
            model, x, y, u, instance_arrival_rate, instances_per_app_site = initialize_model(
                sites, pricing, opti_pref, single_app_dict, remaining_capacities, theta
            )
            run_optimization(model)

            if model.status == GRB.OPTIMAL:
                objval_min += model.objVal
                for i in sites:
                    remaining_capacities[i] -= sum(instances_per_app_site[app][i].X * applications[app]["slots"]
                                                for app in single_app_dict)
            else:
                print(f"No optimal solution found for permutation {ordering_apps_final}.")
                break


# ---------- Final results and placement plan generation ----------
    print("====================================================================================================")
    print(f"Evaluated {pi} permutations with initial SLOs in {time_permutations:.2f} seconds")
    if not degraded:
        print("No degradation needed.")
    else:
        print("Degradation applied:")
        print(f"Degraded priority levels {', '.join(map(str, priority_levels))} with applications:")
        for app in slt_apps:
            if app in fully_degraded_apps:
                print(f"  - '{app}' fully degraded to its SLT {applications[app]['slt']} ms.")
            else:
                print(f"  - '{app}' degraded {gap_slo[app]} ms to its new SLO {applications[app]['slo']} ms.")

    print(f"Permutation {ordering_apps_final} with minimum objective value: {objval_min}")
    print(f"Total time cost: {ending_time - starting_time:.2f} seconds")
    print("====================================================================================================")

    placement_plan_json(universal_placement_plan, exp_round_index)


if __name__ == "__main__":
    main()