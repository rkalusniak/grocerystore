#!/usr/bin/env python
# coding: utf-8

import sys
import argparse
from datetime import datetime
import math

import pandas as pd
from numpy.random import default_rng
import simpy
from pathlib import Path


class GroceryStore(object):
    def __init__(self, env, num_carts, num_butchers, num_cashiers,
                 mean_interarrival_time, pct_need_butcher,
                 get_cart_max,
                 produce_mean, produce_sd,
                 butcher_mean, butcher_sd,
                 butcher_time_mean, butcher_time_sigma,
                 pantry_mean, pantry_sd,
                 pick_time_mean, pick_time_sd,
                 cashier_time_mean, cashier_time_sigma,
                 rg
                 ):
        """
        Primary class that encapsulates grocery store resources and flow logic.

        The detailed customer flow logic is in the buy_groceries() method of this class. Also,
        the run_store() function is a run() method in this class. The only customer information
        that gets passed in to any class methods is the customer id (int).

        Parameters
        ----------
        env
        num_carts
        num_butchers
        num_cashiers
        mean_interarrival_time
        pct_need_butcher
        get_cart_max
        produce_mean
        produce_sd
        butcher_mean
        butcher_sd
        butcher_time_mean
        butcher_time_sigma
        pantry_mean
        pantry_sd
        pick_time_mean
        pick_time_sd
        cashier_time_mean
        cashier_time_sigma
        rg
        """

        # Simulation environment and random number generator
        self.env = env
        self.rg = rg

        # Create list to hold timestamps dictionaries (one per patient)
        self.timestamps_list = []
        # Create lists to hold occupancy tuples (time, occ)
        self.store_occupancy_list = [(0.0, 0.0)]
        self.checkout_occupancy_list = [(0.0, 0.0)]

        # Create SimPy resources
        self.cart = simpy.Resource(env, num_carts)
        self.butcher = simpy.Resource(env, num_butchers)
        self.cashier = simpy.Resource(env, num_cashiers)

        # Initialize the patient flow related attributes
        self.mean_interarrival_time = mean_interarrival_time
        self.pct_need_butcher = pct_need_butcher
        self.get_cart_max = get_cart_max
        self.produce_mean = produce_mean
        self.produce_sd = produce_sd
        self.butcher_mean = butcher_mean
        self.butcher_sd = butcher_sd
        self.butcher_time_mean = butcher_time_mean
        self.butcher_time_sigma = butcher_time_sigma
        self.pantry_mean = pantry_mean
        self.pantry_sd = pantry_sd
        self.pick_time_mean = pick_time_mean
        self.pick_time_sd = pick_time_sd
        self.cashier_time_mean = cashier_time_mean
        self.cashier_time_sigma = cashier_time_sigma


    #Assume that each cutomer has one pick_time. This is the the time they take to select one item
    def pick_time(self):
        pick_time = self.rg.normal(self.pick_time_mean, self.pick_time_sd)
        return pick_time

    #Get count of items
    def num_produce(self):
        return max(round(self.rg.normal(self.produce_mean, self.produce_sd), 0), 0)

    def num_pantry(self):
        return max(round(self.rg.normal(self.pantry_mean, self.pantry_sd), 0), 0)

    def num_butcher(self):
        if self.rg.random() < self.pct_need_butcher:
            num_butcher = max(round(self.rg.normal(self.butcher_mean, self.butcher_sd), 0), 0)
        else:
            num_butcher = 0
        return num_butcher

    def num_items(self):
        return self.num_produce() + self.num_butcher() + self.num_pantry()

    # Create process methods
    def get_cart(self):
        yield self.env.timeout(self.rg.uniform(0, self.get_cart_max))

    def pick_produce(self):
        yield self.env.timeout(self.pick_time() * self.num_produce())

    # There is the time and number of items to pick as well as the time it takes the butcher to pull the order.
    def pick_butcher(self):
        butcher_time = self.rg.lognormal(mean=self.butcher_time_mean, sigma=self.butcher_time_sigma)
        yield self.env.timeout(self.num_butcher() * (self.pick_time() + butcher_time))

    def pick_pantry(self):
        yield self.env.timeout(self.pick_time() * self.num_pantry())

    # The time to check out is the number of items times the scanning time per item.
    def check_out(self):
        cashier_time = self.rg.lognormal(mean=self.cashier_time_mean, sigma=self.cashier_time_sigma)
        yield self.env.timeout(self.num_items() * cashier_time)

    def buy_groceries(self, customer, quiet):
        """Defines the sequence of steps traversed by customers.

        Also capture a bunch of timestamps to make it easy to compute various system
        performance measures such as customer waiting times, queue sizes, and resource utilization.
        """
        # Customer arrives to store - note the arrival time
        arrival_ts = self.env.now

        # Update store occupancy - increment by 1
        self.store_occupancy_list.append((self.env.now, self.store_occupancy_list[-1][1] + 1))

        # Request a cart
        # By using request() in a context manager, we'll automatically release the resource when done
        with self.cart.request() as cart_request:
            yield cart_request
            # Now that we have a cart, record time to pullout/walk in and record time.
            got_cart_ts = self.env.now
            yield self.env.process(self.get_cart())

            # Spend time picking produce items. No resources to request
            start_produce_ts = self.env.now
            yield self.env.process(self.pick_produce())
            finish_produce_ts = self.env.now

            #If the person is going to butcher request resource and release when done.
            if self.num_butcher() > 1:
                with self.butcher.request() as request:
                    yield request
                    got_butcher_ts = self.env.now
                    yield self.env.process(self.pick_butcher())
                    release_butcher_ts = self.env.now
            else:
                got_butcher_ts = pd.NA
                release_butcher_ts = pd.NA

            # Spend time picking pantry items. No resources to request
            start_pantry_ts = self.env.now
            yield self.env.process(self.pick_pantry())
            finish_pantry_ts = self.env.now

            # Request a cashier and check out.
            # Request cashier
            with self.cashier.request() as request:
                if not quiet:
                    print(f"Customer {customer} requests cashier at time {self.env.now}")

                # Request filled
                yield request
                got_cashier_ts = self.env.now

                # Time wait for cashier and print results
                q_time = got_cashier_ts - finish_pantry_ts
                if not quiet:
                    print(f"Customer {customer} gets cashier at time {self.env.now} (waited {q_time:.1f} minutes)")

                # Update check out occupancy - increment by 1
                self.checkout_occupancy_list.append((self.env.now, self.checkout_occupancy_list[-1][1] + 1))

                # Perform the checkout
                yield self.env.process(self.check_out())
                exit_system_ts = self.env.now

                # Update check out occupancy - decrement by 1
                self.checkout_occupancy_list.append((self.env.now, self.checkout_occupancy_list[-1][1] - 1))

                # Update store occupancy - decrement by 1
                self.store_occupancy_list.append((self.env.now, self.store_occupancy_list[-1][1] - 1))

        #Leave store
        if not quiet:
            print(f"Customer {customer} released cashier and exited store at time {self.env.now}")


        # Create dictionary of timestamps
        timestamps = {'customer_id': customer,
                      'num_items': self.num_items(),
                      'arrival_ts': arrival_ts,
                      'got_cart_ts': got_cart_ts,
                      'start_produce_ts': start_produce_ts,
                      'finish_produce_ts': finish_produce_ts,
                      'got_butcher_ts': got_butcher_ts,
                      'release_butcher_ts': release_butcher_ts,
                      'start_pantry_ts': start_pantry_ts,
                      'finish_pantry_ts': finish_pantry_ts,
                      'got_cashier_ts': got_cashier_ts,
                      'cashier_speed': (exit_system_ts - got_cashier_ts)/self.num_items(),
                      'exit_system_ts': exit_system_ts}

        self.timestamps_list.append(timestamps)

    def run(self, stoptime=simpy.core.Infinity, max_arrivals=simpy.core.Infinity, quiet=False):
        """
        Run the store for a specified amount of time or after generating a maximum number of customers.

        Parameters
        ----------
        stoptime : float
        max_arrivals : int
        quiet : bool

        Yields
        -------
        Simpy environment timeout
        """

        # Create a counter to keep track of number of customers generated and to serve as unique customer id
        customer = 0

        # Loop for generating patients
        while self.env.now < stoptime and customer < max_arrivals:
            # Generate next interarrival time
            iat = self.rg.exponential(self.mean_interarrival_time)

            # This process will now yield to a 'timeout' event. This process will resume after iat time units.
            yield self.env.timeout(iat)

            # New customer generated = update counter of customer
            customer += 1

            if not quiet:
                print(f"Customer {customer} created at time {self.env.now}")

            # Register buy_groceries process for the new customer
            self.env.process(self.buy_groceries(customer, quiet))

        print(f"{customer} customers processed.")


def compute_durations(timestamp_df):
    """Compute time durations of interest from timestamps dataframe and append new cols to dataframe"""

    timestamp_df['wait_for_cart'] = timestamp_df.loc[:, 'got_cart_ts'] - timestamp_df.loc[:, 'arrival_ts']
    timestamp_df['produce_time'] = timestamp_df.loc[:, 'finish_produce_ts'] - timestamp_df.loc[:, 'start_produce_ts']
    timestamp_df['wait_for_butcher'] = timestamp_df.loc[:, 'got_butcher_ts'] - timestamp_df.loc[:, 'finish_produce_ts']
    timestamp_df['butcher_time'] = timestamp_df.loc[:, 'release_butcher_ts'] - timestamp_df.loc[:, 'got_butcher_ts']
    timestamp_df['pantry_time'] = timestamp_df.loc[:, 'finish_pantry_ts'] - timestamp_df.loc[:, 'start_pantry_ts']
    timestamp_df['wait_for_cashier'] = timestamp_df.loc[:, 'got_cashier_ts'] - timestamp_df.loc[:, 'finish_produce_ts']
    timestamp_df['checkout_time'] = timestamp_df.loc[:, 'exit_system_ts'] - timestamp_df.loc[:, 'got_cashier_ts']
    timestamp_df['time_in_system'] = timestamp_df.loc[:, 'exit_system_ts'] - timestamp_df.loc[:, 'arrival_ts']
    timestamp_df['time_in_system_peritem'] = timestamp_df.loc[:, 'time_in_system'] / timestamp_df.loc[:, 'num_items']

    return timestamp_df


def simulate(arg_dict, rep_num):
    """

    Parameters
    ----------
    arg_dict : dict whose keys are the input args
    rep_num : int, simulation replication number

    Returns
    -------
    Nothing returned but numerous output files written to ``args_dict[output_path]``

    """

    # Create a random number generator for this replication
    seed = arg_dict['seed'] + rep_num - 1
    rg = default_rng(seed=seed)

    # Resource capacity levels
    num_carts = arg_dict['num_carts']
    num_butchers = arg_dict['num_butchers']
    num_cashiers = arg_dict['num_cashiers']

    # Initialize the patient flow related attributes
    customer_arrival_rate = arg_dict['customer_arrival_rate']
    mean_interarrival_time = 1.0 / (customer_arrival_rate / 60.0)

    pct_need_butcher = arg_dict['pct_need_butcher']
    get_cart_max = arg_dict['get_cart_max']
    produce_mean = arg_dict['produce_mean']
    produce_sd = arg_dict['produce_sd']
    butcher_mean = arg_dict['butcher_mean']
    butcher_sd = arg_dict['butcher_sd']
    butcher_time_mean = arg_dict['butcher_time_mean']
    butcher_time_sigma = arg_dict['butcher_time_sigma']
    pantry_mean = arg_dict['pantry_mean']
    pantry_sd = arg_dict['pantry_sd']
    pick_time_mean = arg_dict['pick_time_mean']
    pick_time_sd = arg_dict['pick_time_sd']
    cashier_time_mean = arg_dict['cashier_time_mean']
    cashier_time_sigma = arg_dict['cashier_time_sigma']

    # Other parameters
    stoptime = arg_dict['stoptime']  # No more arrivals after this time
    quiet = arg_dict['quiet']
    scenario = arg_dict['scenario']

    # Run the simulation
    env = simpy.Environment()

    # Create a store to simulate
    store = GroceryStore(env, num_carts, num_butchers, num_cashiers,
                         mean_interarrival_time, pct_need_butcher,
                         get_cart_max,
                         produce_mean, produce_sd,
                         butcher_mean, butcher_sd,
                         butcher_time_mean, butcher_time_sigma,
                         pantry_mean, pantry_sd,
                         pick_time_mean, pick_time_sd,
                         cashier_time_mean, cashier_time_sigma,
                         rg)


    # Initialize and register the run generator function
    env.process(store.run(stoptime=stoptime, quiet=quiet))

    # Launch the simulation
    env.run()

    # Create output files and basic summary stats
    if len(arg_dict['output_path']) > 0:
        output_dir = Path.cwd() / arg_dict['output_path']
    else:
        output_dir = Path.cwd()

    # Create paths for the output logs
    store_customer_log_path = output_dir / f'store_customer_log_{scenario}_{rep_num}.csv'
    store_occupancy_df_path = output_dir / f'store_occupancy_{scenario}_{rep_num}.csv'
    checkout_occupancy_df_path = output_dir / f'checkout_occupancy_{scenario}_{rep_num}.csv'

    # Create patient log dataframe and add scenario and rep number cols
    store_customer_log_df = pd.DataFrame(store.timestamps_list)
    store_customer_log_df['scenario'] = scenario
    store_customer_log_df['rep_num'] = rep_num

    # Reorder cols to get scenario and rep_num first
    num_cols = len(store_customer_log_df.columns)
    new_col_order = [-2, -1]
    new_col_order.extend([_ for _ in range(0, num_cols-2)])
    store_customer_log_df = store_customer_log_df.iloc[:, new_col_order]

    # Compute durations of interest for patient log
    store_customer_log_df = compute_durations(store_customer_log_df)

    # Create occupancy log dataframes and add scenario and rep number cols
    store_occupancy_df = pd.DataFrame(store.store_occupancy_list, columns=['ts', 'occ'])
    store_occupancy_df['scenario'] = scenario
    store_occupancy_df['rep_num'] = scenario
    num_cols = len(store_occupancy_df.columns)
    new_col_order = [-2, -1]
    new_col_order.extend([_ for _ in range(0, num_cols - 2)])
    store_occupancy_df = store_occupancy_df.iloc[:, new_col_order]

    checkout_occupancy_df = pd.DataFrame(store.checkout_occupancy_list, columns=['ts', 'occ'])
    checkout_occupancy_df['scenario'] = scenario
    checkout_occupancy_df['rep_num'] = scenario
    num_cols = len(checkout_occupancy_df.columns)
    new_col_order = [-2, -1]
    new_col_order.extend([_ for _ in range(0, num_cols - 2)])
    checkout_occupancy_df = checkout_occupancy_df.iloc[:, new_col_order]

    # Export logs to csv
    store_customer_log_df.to_csv(store_customer_log_path, index=False)
    store_occupancy_df.to_csv(store_occupancy_df_path, index=False)
    checkout_occupancy_df.to_csv(checkout_occupancy_df_path, index=False)

    # Note simulation end time
    end_time = env.now
    print(f"Simulation replication {rep_num} ended at time {end_time}")


def process_sim_output(csvs_path, scenario, performance_measures):
    """

    Parameters
    ----------
    csvs_path : Path object for location of simulation output patient log csv files
    scenario : str

    Returns
    -------
    Dict of dicts

    Keys are:

    'customer_log_rep_stats' --> Contains dataframes from describe on group by rep num. Keys are perf measures.
    'customer_log_ci' -->        Contains dictionaries with overall stats and CIs. Keys are perf measures.
    """

    dest_path = csvs_path / f"consolidated_store_customer_log_{scenario}.csv"

    sort_keys = ['scenario', 'rep_num']

    # Create empty dict to hold the DataFrames created as we read each csv file
    dfs = {}

    # Loop over all the csv files
    for csv_f in csvs_path.glob(f'store_customer_log_{scenario}*.csv'):
        # Split the filename off from csv extension. We'll use the filename
        # (without the extension) as the key in the dfs dict.
        fstem = csv_f.stem

        # Read the next csv file into a pandas DataFrame and add it to
        # the dfs dict.
        df = pd.read_csv(csv_f)
        dfs[fstem] = df

    # Use pandas concat method to combine the file specific DataFrames into
    # one big DataFrame.
    customer_log_df = pd.concat(dfs)

    # Since we didn't try to control the order in which the files were read,
    # we'll sort the final DataFrame in place by the specified sort keys.
    customer_log_df.sort_values(sort_keys, inplace=True)

    # Export the final DataFrame to a csv file. Suppress the pandas index.
    customer_log_df.to_csv(dest_path, index=False)

    # Compute summary statistics for several performance measures
    customer_log_stats = summarize_customer_log(customer_log_df, scenario, performance_measures)

    # Now delete the individual replication files - *******Removed this for running this program separately
    # for csv_f in csvs_path.glob('store_customer_log_*.csv'):
    #     csv_f.unlink()

    return customer_log_stats


def summarize_customer_log(customer_log_df, scenario, performance_measures):
    """

    Parameters
    ----------
    customer_log_df : DataFrame created by process_sim_output
    scenario : str

    Returns
    -------
    Dict of dictionaries - See comments below
    """

    # Create empty dictionaries to hold computed results
    customer_log_rep_stats = {}  # Will store dataframes from describe on group by rep num. Keys are perf measures.
    customer_log_ci = {}         # Will store dictionaries with overall stats and CIs. Keys are perf measures.
    customer_log_stats = {}      # Container dict returned by this function containing the two previous dicts.

    # Loop over performance measures from input
    for pm in performance_measures:
        # Compute descriptive stats for each replication and store dataframe in dict
        customer_log_rep_stats[pm] = customer_log_df.groupby(['rep_num'])[pm].describe()
        # Compute across replication stats
        n_samples = customer_log_rep_stats[pm]['mean'].count()
        mean_mean = customer_log_rep_stats[pm]['mean'].mean()
        sd_mean = customer_log_rep_stats[pm]['mean'].std()
        ci_95_lower = mean_mean - 1.96 * sd_mean / math.sqrt(n_samples)
        ci_95_upper = mean_mean + 1.96 * sd_mean / math.sqrt(n_samples)
        # Store cross replication stats as dict in dict
        customer_log_ci[pm] = {'n_samples': n_samples, 'mean_mean': mean_mean, 'sd_mean': sd_mean,
                               'ci_95_lower': ci_95_lower, 'ci_95_upper': ci_95_upper}

    customer_log_stats['scenario'] = scenario
    customer_log_stats['customer_log_rep_stats'] = customer_log_rep_stats
    # Convert the final summary stats dict to a DataFrame
    customer_log_stats['customer_log_ci'] = pd.DataFrame(customer_log_ci)

    return customer_log_stats


def process_command_line():
    """
    Parse command line arguments

    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    Return a Namespace representing the argument list.
    """

    # Create the parser
    parser = argparse.ArgumentParser(prog='grocerystore_model',
                                     description='Run grocery store simulation')

    # Add arguments
    parser.add_argument(
        "--config", type=str, default=None,
        help="Configuration file containing input parameter arguments and values"
    )

    parser.add_argument("--customer_arrival_rate", default=150, help="customers per hour",
                        type=float)

    parser.add_argument("--num_carts", default=300, help="number of carts",
                        type=int)

    parser.add_argument("--num_butchers", default=1, help="number of butchers",
                        type=int)

    parser.add_argument("--num_cashiers", default=12, help="number of cashiers",
                        type=int)

    parser.add_argument("--pct_need_butcher", default=0.2,
                        help="percent of customers that need the butcher (default = 0.2)",
                        type=float)

    parser.add_argument("--get_cart_max", default=2.0,
                        help="Max time (mins) to get cart. Uniform distribution between zero and max. (default = 2)",
                        type=float)

    parser.add_argument("--produce_mean", default=7.0,
                        help="Mean number of produce items (default = 7)",
                        type=float)

    parser.add_argument("--produce_sd", default=2.0,
                        help="Standard deviation number of produce items (default = 2)",
                        type=float)

    parser.add_argument("--butcher_mean", default=2.0,
                        help="Mean number of butcher items (default = 2)",
                        type=float)

    parser.add_argument("--butcher_sd", default=0.2,
                        help="Standard deviation number of butcher items (default = 0.2)",
                        type=float)

    parser.add_argument("--butcher_time_mean", default=0,
                        help="Mean time (min) per butcher item (default = 0)",
                        type=float)

    parser.add_argument("--butcher_time_sigma", default=0.1,
                        help="Sigma time (min) per butcher items (default = 0.1)",
                        type=float)

    parser.add_argument("--pantry_mean", default=10.0,
                        help="Mean number of pantry items (default = 10.0)",
                        type=float)

    parser.add_argument("--pantry_sd", default=2.0,
                        help="Standard deviation number of pantry items (default = 2.0)",
                        type=float)

    parser.add_argument("--pick_time_mean", default=2,
                        help="Mean time (min) to pick item in produce, butcher, or pantry (default = 2)",
                        type=float)

    parser.add_argument("--pick_time_sd", default=0.4,
                        help="Standard deviation time (min) to pick item in produce, butcher, or pantry (default = 0.4)",
                        type=float)

    parser.add_argument("--cashier_time_mean", default=-2,
                        help="Mean time (min) to checkout per item (default = -2)",
                        type=float)

    parser.add_argument("--cashier_time_sigma", default=1,
                        help="Sigma time (min) to checkout per item (default = 1)",
                        type=float)

    parser.add_argument("--scenario", default=datetime.now().strftime("%Y.%m.%d.%H.%M."),
                        help="Appended to output filenames.",
                        type=str)

    parser.add_argument("--stoptime", default=1080,
                        help="time that simulation stops (default = 1080)",
                        type=float)

    parser.add_argument("--num_reps", default=1,
                        help="number of simulation replications (default = 1)",
                        type=int)

    parser.add_argument("--seed", default=8842,
                        help="random number generator seed (default = 8842)",
                        type=int)

    parser.add_argument("--output_path", default="output",
                        help="location for output file writing",
                        type=str)

    parser.add_argument("--quiet", action='store_true',
                        help="If True, suppresses output messages (default=False)")

    # do the parsing
    args = parser.parse_args()

    if args.config is not None:
        # Read inputs from config file
        with open(args.config, "r") as fin:
            args = parser.parse_args(fin.read().split())

    return args


def main():

    args = process_command_line()
    print(args)

    #Load inputs from config file
    num_reps = args.num_reps
    scenario = args.scenario

    if len(args.output_path) > 0:
        output_dir = Path.cwd() / args.output_path
    else:
        output_dir = Path.cwd()

    # Delete any previous senario documents. This is important if number of reps changes
    del_files = ['checkout_occupancy', 'store_occupancy',
                'store_customer_log', 'consolidated_store_log']
    for del_f in del_files:
        for csv_f in output_dir.glob(f'{del_f}{scenario}*.csv'):
            csv_f.unlink()

    # Main simulation replication loop
    for i in range(1, num_reps + 1):
        simulate(vars(args), i)

    # Consolidate the patient logs and compute summary stats
    # Create list of performance measures for looping over
    performance_measures = ['wait_for_cart', 'wait_for_butcher', 'wait_for_cashier',
                            'time_in_system', 'time_in_system_peritem', 'num_items']
    customer_log_stats = process_sim_output(output_dir, scenario, performance_measures)
    print(f"\nScenario: {scenario}")
    pd.set_option("display.precision", 3)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    print(customer_log_stats['customer_log_rep_stats'])
    print(customer_log_stats['customer_log_ci'])


if __name__ == '__main__':
    main()


