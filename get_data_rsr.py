import torch
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

from torch_geometric.data import Data
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from statsmodels.tsa.seasonal import STL

def get_data_rsr():

    dir = '/home/study/IdeaProjects/Graph-Machine-Learning/Temporal_RSR/data'

    """
    COPIED FROM THE PAPER
    source code: https://github.com/fulifeng/Temporal_Relational_Stock_Ranking
    """
    def load_EOD_data(data_path, market_name, tickers, steps=1):
        eod_data = []
        masks = []
        ground_truth = []
        base_price = []

        # Determine the expected number of rows based on the first ticker's data
        first_ticker_path = os.path.join(data_path, market_name + '_' + tickers[0] + '_1.csv')
        try:
            first_df = pd.read_csv(first_ticker_path, header=None)
            num_days = first_df.shape[0] - (1 if market_name == 'NASDAQ' else 0) # Remove last row for NASDAQ
            num_features = first_df.shape[1] - 1 # Exclude the date column
        except Exception as e:
            print(f"Error reading first ticker file {first_ticker_path}: {e}")
            return None, None, None, None

        eod_data = np.zeros([len(tickers), num_days, num_features], dtype=np.float32)
        masks = np.ones([len(tickers), num_days], dtype=np.float32)
        ground_truth = np.zeros([len(tickers), num_days], dtype=np.float32) # We're not using this one
        base_price = np.zeros([len(tickers), num_days], dtype=np.float32)

        for index, ticker in enumerate(tickers):
            if index % 50 == 0:
                print(f"Processed [{index}/{tickers.shape[0]}] tickers")
            single_EOD_path = os.path.join(data_path, market_name + '_' + ticker + '_1.csv')

            try:
                single_df = pd.read_csv(single_EOD_path, header=None)
                if market_name == 'NASDAQ':
                    single_df = single_df[:-1] # remove the last day since lots of missing data

                # Handle missing values (-1234)
                single_EOD = single_df.values
                mask_row_indices, mask_col_indices = np.where(np.abs(single_EOD + 1234) < 1e-8)
                single_EOD[mask_row_indices, mask_col_indices] = 1.1 # Replace missing values

                # Update masks based on missing closing price
                missing_close_indices = np.where(np.abs(single_EOD[:, -1] + 1234) < 1e-8)[0]
                masks[index, missing_close_indices] = 0.0

                eod_data[index, :, :] = single_EOD[:, 1:] # Exclude date column
                base_price[index, :] = single_EOD[:, -1]

            except Exception as e:
                print(f"Error reading ticker file {single_EOD_path}: {e}")
                # Mark all days for this ticker as invalid if file reading fails
                masks[index, :] = 0.0


        print('eod data shape:', eod_data.shape)
        return eod_data, masks, ground_truth, base_price

    """
    COPIED FROM THE PAPER
    source code: https://github.com/fulifeng/Temporal_Relational_Stock_Ranking
    """
    def load_relation_data(relation_file):
        relation_encoding = np.load(relation_file)
        print('relation encoding shape:', relation_encoding.shape)
        rel_shape = [relation_encoding.shape[0], relation_encoding.shape[1]]
        mask_flags = np.equal(np.zeros(rel_shape, dtype=int),
                              np.sum(relation_encoding, axis=2))
        mask = np.where(mask_flags, np.ones(rel_shape) * -1e9, np.zeros(rel_shape))
        return relation_encoding, mask

    industry_encodings, industry_mask = load_relation_data(dir+'/relation/sector_industry/NASDAQ_industry_relation.npy')
    wiki_encodings, wiki_mask = load_relation_data(dir+'/relation/wikidata/NASDAQ_wiki_relation.npy')

    # Load company names
    tickers = np.loadtxt(dir+'/NASDAQ_tickers.csv', dtype=str)
    print('tickers shape (# of companies):', tickers.shape)

    eod_data, eod_masks, eod_ground_truth, eod_base_price = load_EOD_data(dir+"/2013-01-01", "NASDAQ", tickers)

    # Use subset of data for the experiments
    n_companies = 150

    wiki_encodings = wiki_encodings[:n_companies, :n_companies, :]
    wiki_mask = wiki_mask[:n_companies, :n_companies]
    industry_encodings = industry_encodings[:n_companies, :n_companies, :]
    industry_mask = industry_mask[:n_companies, :n_companies]

    eod_data, eod_masks, eod_ground_truth, eod_base_price = load_EOD_data(dir+"/2013-01-01", "NASDAQ", tickers[:n_companies])
    # ============================================================================
    # Data Preparation Functions
    # ============================================================================

    def build_adjacency_matrix(industry_encodings, industry_mask, wiki_encodings, wiki_mask, device):
        """
        Build normalized adjacency matrix from relation encodings and masks

        Args:
            industry_encodings: [num_companies, num_companies, num_relation_types]
            industry_mask: [num_companies, num_companies] (-1e9 for no relation, 0 for valid)
            wiki_encodings: [num_companies, num_companies, num_relation_types]
            wiki_mask: [num_companies, num_companies]

        Returns:
            adjacency_matrix: [num_companies, num_companies] - normalized adjacency
        """
        # Combine relation encodings by summing across relation types
        industry_adj = torch.sum(industry_encodings, dim=-1)  # [companies, companies]
        wiki_adj = torch.sum(wiki_encodings, dim=-1)

        # Combine both relation types
        combined_adj = industry_adj + wiki_adj

        # Apply masks: where mask is -1e9 (no relation), set adjacency to 0
        combined_mask = industry_mask + wiki_mask
        combined_adj = torch.where(combined_mask < -1e8, torch.zeros_like(combined_adj), combined_adj)

        # Normalize: row-wise normalization (each row sums to 1)
        row_sums = combined_adj.sum(dim=1, keepdim=True)
        adjacency_matrix = combined_adj / (row_sums + 1e-8)

        return adjacency_matrix.to(device)


    def prepare_data(eod_data, masks, base_price, device, window_size=20, prediction_horizon=1):
        """
        Create sliding windows for time series prediction with mask handling

        Args:
            eod_data: [num_companies, num_days, num_features]
            masks: [num_companies, num_days] - 1.0 for valid, 0.0 for missing
            base_price: [num_companies, num_days] - closing price of stock
            window_size: Number of historical days to use as input
            prediction_horizon: Number of days ahead to predict (usually 1)

        Returns:
            X: Input windows [num_samples, num_companies, window_size, num_features]
            y: Target returns [num_samples, num_companies, prediction_horizon]
            sample_masks: Valid sample indicators [num_samples, num_companies]
        """
        num_companies, num_days, num_features = eod_data.shape
        num_samples = num_days - window_size - prediction_horizon + 1

        X = torch.zeros(num_samples, num_companies, window_size, num_features, device=device)
        y = torch.zeros(num_samples, num_companies, prediction_horizon, device=device)
        sample_masks = torch.zeros(num_samples, num_companies, device=device)

        for i in range(num_samples):
            X[i] = eod_data[:, i:i+window_size, :]
            y[i, :, 0] = base_price[:, i+window_size+prediction_horizon-1]

            # A sample is valid if all days in the window AND the target day are valid
            window_valid = masks[:, i:i+window_size].min(dim=1)[0]  # [num_companies]
            target_valid = masks[:, i+window_size+prediction_horizon-1]
            sample_masks[i] = window_valid * target_valid

        return X, y, sample_masks

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Load data
    num_companies = 150 # Change if using subset
    num_days = 1245
    num_features = 5

    eod_data = torch.tensor(eod_data)
    masks = torch.tensor(eod_masks)
    price_prediction = torch.tensor(eod_base_price) # We predict stock price

    # Relation data
    industry_encodings = torch.tensor(industry_encodings)
    industry_mask = torch.tensor(industry_mask)
    wiki_encodings = torch.tensor(wiki_encodings)
    wiki_mask = torch.tensor(wiki_mask)

    print(f"EOD data shape: {eod_data.shape}")
    print(f"Masks shape: {masks.shape}")
    print(f"Ground truth shape: {price_prediction.shape}")
    print(f"Industry encodings shape: {industry_encodings.shape}")

    # Build adjacency matrix from relations
    adjacency_matrix = build_adjacency_matrix(
        industry_encodings, industry_mask,
        wiki_encodings, wiki_mask,
        device=device
    )
    print(f"Adjacency matrix shape: {adjacency_matrix.shape}")

    # Prepare temporal data with masks
    window_size = 20  # Use window_size days of history
    X_train, y_train, train_masks = prepare_data(
        eod_data, masks, price_prediction,
        window_size=window_size,
        device=device,
        prediction_horizon=1
    )
    print(f"Training data: X={X_train.shape}, y={y_train.shape}, masks={train_masks.shape}")
    # num of days x


    #Convert to STGAT format


    def adjacency_to_edges(adjacency_matrix):
        """Convert adjacency matrix to edge_index and edge_weight"""
        adj_np = adjacency_matrix.cpu().numpy()
        rows, cols = np.where(adj_np > 0)
        edge_weights = adj_np[rows, cols]
        edge_index = torch.tensor(np.stack([rows, cols]), dtype=torch.long) # todo check if this is the formatting that is needed
        edge_weight = torch.tensor(edge_weights, dtype=torch.float32).view(-1, 1) # todo check if this is correctly formatted for the model
        return edge_index, edge_weight

    # Convert adjacency matrix to edge representation
    edge_index, edge_weight = adjacency_to_edges(adjacency_matrix) # Todo I think STGAT does calculate graph per window, but idk how that works here or what we should do with that
    # todo at the moment I just use the adhacency matrix, and use the same graph every data object

    # Todo a problem is that now nodes have an edge to itself, almost all of them. Is that what we want? Or what is needed?
    # Create list of Data objects for STGAT
    data_list = []
    num_samples = X_train.shape[0]

    for i in range(num_samples):
        # Get sequence for this sample: [num_companies, window_size, num_features]
        sequence = X_train[i][:, :, -1:]  # [150, 20, 1]
        target = y_train[i]  # [150, 1]
        mask = train_masks[i]  # [150] # todo we still need to use the mask and add it to the data object and change the loss function

        sequence_cpu = sequence.cpu().numpy()

        # Initialize arrays to store STL results for all companies
        t_features = np.zeros((num_companies, window_size))
        s_features = np.zeros((num_companies, window_size))
        r_features = np.zeros((num_companies, window_size))

        # Apply STL to each company's time series separately
        for company_idx in range(num_companies):
            # Extract 1D time series for this company: [window_size]
            company_series = sequence_cpu[company_idx, :, 0]  # [20]

            # Apply STL decomposition
            stl = STL(company_series, period=20, robust=True)
            result = stl.fit()

            # Store results
            t_features[company_idx] = result.trend
            s_features[company_idx] = result.seasonal
            r_features[company_idx] = result.resid

        # Convert back to tensors
        t_features = torch.tensor(t_features, dtype=torch.float32)
        s_features = torch.tensor(s_features, dtype=torch.float32)
        r_features = torch.tensor(r_features, dtype=torch.float32)

        # Todo check this part, idk what the x features should be , for now i use the s features
        x_features = s_features

        # Create Data object
        data = Data(
            x=x_features.float(),
            t=t_features.float(),
            s=s_features.float(),
            r=r_features.float(),
            edge_index=edge_index.to(device),
            edge_weight=edge_weight.to(device),
            shouchujia=target.squeeze().float(),
        )

        data.num_nodes = num_companies
        data_list.append(data)

    print(f"\nGenerated {len(data_list)} samples")
    print(f"First sample shapes:")
    print(f"  x: {data_list[0].x.shape}")
    print(f"  t: {data_list[0].t.shape}")
    print(f"  s: {data_list[0].s.shape}")
    print(f"  r: {data_list[0].r.shape}")
    print(f"  edge_index: {data_list[0].edge_index.shape}")
    print(f"  edge_weight: {data_list[0].edge_weight.shape}")
    print(f"  shouchujia: {data_list[0].shouchujia.shape}")

    return data_list, num_features, 0, 0
