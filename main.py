# coding=utf-8
import torch
import os
import sys
import argparse
import logging
import datetime
import json

"""
Goal: Program Main - Production-ready trading application.
Authors: Thibaut Théate and Damien Ernst
Institution: University of Liège
"""

###############################################################################
################################### Imports ###################################
###############################################################################

from tradingSimulator import TradingSimulator

###############################################################################
############################ Logging Configuration ############################
###############################################################################

def setup_logging(log_level=logging.INFO):
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    if not os.path.exists("logs"):
        os.makedirs("logs")
        
    # Configure logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/trading_{timestamp}.log"
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

###############################################################################
############################### Helper Functions ##############################
###############################################################################

def save_run_config(args):
    """Save the run configuration to a JSON file"""
    if not os.path.exists("run_configs"):
        os.makedirs("run_configs")
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config_filename = f"run_configs/config_{timestamp}.json"
    
    # Convert namespace to dictionary
    config_dict = vars(args)
    
    with open(config_filename, 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    return config_filename

###############################################################################
##################################### MAIN ####################################
###############################################################################

if(__name__ == '__main__'):

    # # Train a new model with custom parameters
    # python main.py -strategy TDQN -stock Google -mode train -save_model models/tdqn_google.pt -episodes 100 -start_date 2015-1-1 -verbose

    # # Continue training from a saved model
    # python main.py -strategy TDQN -stock Google -mode train -load_model models/tdqn_google.pt -save_model models/tdqn_google_continued.pt -episodes 50

    # # Test a saved model
    # python main.py -strategy TDQN -stock Google -mode test -load_model models/tdqn_google.pt -verbose

    # # Run a comprehensive backtest
    # python main.py -strategy TDQN -mode backtest -load_model models/tdqn_google.pt


    # Set up logging
    logger = setup_logging()
    logger.info("Application starting...")
    
    # Check for CUDA availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Retrieve the parameters sent by the user
    parser = argparse.ArgumentParser(description='Deep Reinforcement Learning for Algorithmic Trading')
    parser.add_argument("-strategy", default='TDQN', type=str, help="Name of the trading strategy")
    parser.add_argument("-stock", default='JPMorgan Chase', type=str, help="Name of the stock (market)")
    parser.add_argument("-load_model", default=None, type=str, help="Path to load a saved model")
    parser.add_argument("-save_model", default=None, type=str, help="Path to save the trained model")
    parser.add_argument("-mode", default="train", type=str, choices=["train", "test", "backtest"], help="Mode: train, test, or backtest")
    parser.add_argument("-episodes", default=5, type=int, help="Number of training episodes")
    parser.add_argument("-start_date", default='2012-1-1', type=str, help="Starting date for training/testing")
    parser.add_argument("-split_date", default='2018-1-1', type=str, help="Splitting date between training/testing")
    parser.add_argument("-end_date", default='2020-1-1', type=str, help="Ending date for testing")
    parser.add_argument("-initial_capital", default=100000, type=float, help="Initial capital for trading")
    parser.add_argument("-transaction_costs", default=0.002, type=float, help="Transaction costs (as percentage)")
    parser.add_argument("-verbose", action="store_true", help="Enable verbose mode")
    parser.add_argument("-no_plot", action="store_true", help="Disable plotting")
    parser.add_argument("-no_render", action="store_true", help="Disable rendering")
    args = parser.parse_args()
    
    # Save the run configuration
    config_file = save_run_config(args)
    logger.info(f"Run configuration saved to {config_file}")
    
    # Create models directory if it doesn't exist
    if args.save_model and not os.path.exists(os.path.dirname(args.save_model)):
        os.makedirs(os.path.dirname(args.save_model))
        logger.info(f"Created directory: {os.path.dirname(args.save_model)}")
    
    # Initialization of the required variables
    simulator = TradingSimulator()
    strategy = args.strategy
    stock = args.stock
    
    # Set default save path if not provided but saving is requested in train mode
    save_path = args.save_model
    if save_path is None and args.mode == "train":
        # Create models directory if it doesn't exist
        if not os.path.exists("models"):
            os.makedirs("models")
        save_path = f"models/{strategy}_{stock}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        logger.info(f"Using default save path: {save_path}")

    try:
        # Training or testing based on the mode
        if args.mode == "train":
            logger.info(f"Training {strategy} on {stock}...")
            if args.load_model:
                logger.info(f"Continuing training from model: {args.load_model}")
            
            # Train the model with the specified number of episodes
            tradingStrategy, trainingEnv, testingEnv = simulator.simulateNewStrategy(
                strategy, 
                stock, 
                startingDate=args.start_date,
                endingDate=args.end_date,
                splitingDate=args.split_date,
                money=args.initial_capital,
                transactionCosts=args.transaction_costs,
                saveStrategy=True if save_path else False,
                custom_save_path=save_path,
                custom_load_path=args.load_model,
                numberOfEpisodes=args.episodes,
                verbose=args.verbose,
                plotTraining=not args.no_plot,
                rendering=not args.no_render
            )
            
            # Display results and analysis
            simulator.displayTestbench()
            simulator.analyseTimeSeries(stock)
            simulator.simulateExistingStrategy(strategy, stock, custom_load_path=save_path)
            simulator.evaluateStrategy(strategy, saveStrategy=True)
            simulator.evaluateStock(stock)
            
            # Log training completion
            logger.info(f"Training completed. Model saved to {save_path if save_path else 'default location'}")
        
        elif args.mode == "test":
            if args.load_model is None:
                logger.error("Testing mode requires a model to load. Please specify -load_model.")
                sys.exit(1)
                
            logger.info(f"Testing {strategy} on {stock} using model: {args.load_model}")
                
            # Test the loaded model
            tradingStrategy, trainingEnv, testingEnv = simulator.simulateExistingStrategy(
                strategy, 
                stock,
                startingDate=args.start_date,
                endingDate=args.end_date,
                splitingDate=args.split_date,
                money=args.initial_capital,
                transactionCosts=args.transaction_costs,
                rendering=not args.no_render,
                custom_load_path=args.load_model
            )
                
            # Display results and analysis
            simulator.displayTestbench()
            simulator.analyseTimeSeries(stock)
            simulator.evaluateStrategy(strategy, saveStrategy=False)
            simulator.evaluateStock(stock)
            
            # Log testing completion
            logger.info("Testing completed.")
        
        elif args.mode == "backtest":
            if args.load_model is None:
                logger.error("Backtesting mode requires a model to load. Please specify -load_model.")
                sys.exit(1)
                
            logger.info(f"Backtesting {strategy} on all stocks using model: {args.load_model}")
            
            # Perform comprehensive backtesting on all stocks
            results = simulator.evaluateStrategy(
                strategy, 
                startingDate=args.start_date,
                endingDate=args.end_date,
                splitingDate=args.split_date,
                money=args.initial_capital,
                transactionCosts=args.transaction_costs,
                verbose=args.verbose,
                rendering=not args.no_render
            )
            
            # Log backtest completion
            logger.info("Backtesting completed.")
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        sys.exit(1)
    
    logger.info("Application completed successfully.")