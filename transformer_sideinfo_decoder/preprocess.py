import numpy as np
import math
import os
"""
@author: ZHU Haoren
modified by XU Mingshi
"""
def transform_cov_image(closes, lag_t):
    matrixs = []
    length = closes.shape[0]  # length: in their dataset T = 3740
    for i in range(length):
        if i < lag_t:
            continue
        else:
            sub_closes = closes[i - lag_t: i,:]  # (lag, N)
            matrix = np.cov(np.transpose(sub_closes))  # (N, N) = (32, 32)
            vec = matrix[np.triu_indices(matrix.shape[0],0)]  # ((N*(N+1))/2)
            matrixs.append(vec)  # (length - lag_t, (N*(N+1))/2) = (3697, 528)
    return np.array(matrixs)


def normalize_cor_image(image, low = 0, upper = 2):
    return (image + 1) * (upper - low)/2 + low


def transform_cor_image(closes, lag_t, normal=False, low = 0, upper = 1):
    # Maintain diagonal
    corr_matrixs = []
    length = closes.shape[0]
    for i in range(length):
        if i < lag_t:
            continue
        else:
            sub_closes = closes[i - lag_t: i,:]
            corr_matrix = np.corrcoef(np.transpose(sub_closes))
            if True in np.isnan(corr_matrix):
                corr_matrixs.append(corr_matrixs[-1])
            else:
                '''
                matrix = np.array([[1,2,3],[4,5,6],[7,8,9]])
                matrix[np.triu_indices(matrix.shape[0],0)]
                >> array([1, 2, 3, 5, 6, 9])
                matrix[np.triu_indices(matrix.shape[0],-1)]
                >> array([1, 2, 3, 4, 5, 6, 8, 9])
                 matrix[np.triu_indices(matrix.shape[0],1)]
                >> array([2, 3, 6])
                '''
                if normal == True:
                    corr_matrix = normalize_cor_image(corr_matrix, low, upper)
                corr_vec = corr_matrix[np.triu_indices(corr_matrix.shape[0],0)]
                corr_matrixs.append(corr_vec)

    return np.array(corr_matrixs)


def final_preprocess(returns, num_asset, lag_t, input_length=10, input_gap=1, output_length=1, \
                     output_gap=1, rebalance=21, split=[0.85, 0.05, 0.1], normal=True, \
                     low=0, upper=1, cov_scale="log", sparse_type = 0,  local_size = 4, lag_size = 5):
    """
    This is the final version of handling preprocessing.

    params:
        returns:      the log returns of assets
        num_asset:    the number of assets
        lag_t:        the time lag to calculate the matrix
        input_length: the length of input sequence
        input_gap:    the distance between input matrix
        output_length:the length of output sequence
        output_gap:   the distance between output matrix
        rebalance:    the distance to predict the future matrix \
        (i.e. the distance between last input matrix and first output matrix)
        split:        the train, valid, test split ratio
        normal:       whether to normalize the correlation
        low:          lower boundary of normalization, used if normal set true
        upper:        upper boundary of normalization, used if normal set true

    returns:
        d_covs: covariance data in shape of [train, valid, test]
        d_cors: correlation data in shape of [train, valid, test]
    """
    # Get cov and cor matrices whole sequence
    # Generate cov data
    print("into final")
    covs = transform_cov_image(returns, lag_t)  # (length - lag_t, N*(N+1))/2)

    if cov_scale == "log":
        print(covs)
        print(covs.shape)

        isnan = np.isnan(covs)  # 判断每个元素是不是nan,返回[False,False,False,False,True]
        print(True in isnan)
        print(np.any(covs <= 0))

        covs = np.log(covs)  # logarithm scaler, should be made as hyperparameters

    # Generate correlation data
    if normal == False:
        cors = transform_cor_image(returns, lag_t)
    else:
        cors = transform_cor_image(returns, lag_t, normal, low, upper)
    '''
    sparse_type:
    0: Linear Sparse Self Attention
    1: Log Sparse Self Attention
    2: LocalAttention + LogSparse Self Attention
    3: Log lagged Sparse Self Attention
    '''
    def get_frame(sequences, input_length=10, input_gap=1, output_length=1, \
                  output_gap=1, rebalance=21, split=[0.85, 0.05, 0.1], sparse_type=0, local_size=local_size, lag_size=lag_size):
        data = []
        shape = sequences.shape  # (3697, 528)
        # Linear Sparse Self Attention
        if sparse_type == 0:
            input_span = (input_length - 1) * input_gap + 1  # default: 9*21 + 1
            print("input_span ", end='')
            print(input_span)
            num_data = shape[0] - input_span - rebalance + 1  # 3487
            # print(input_length) # 10
            # print(input_gap) # 21
            # print(rebalance) # 21
            # print(input_span) # 190
            # print(num_data) # 3487
            # height = int(np.ceil(pow(shape[1], 0.5)))  # 23
            for i in range(num_data):
                # Get w/o padding input frames and input frames idx
                input_frames = []
                frame_idx = []
                for j in range(input_length):  # 10
                    frame_idx.append(i + j * input_gap)
                    input_frames.append(sequences[i + j * input_gap])  # 每次间隔为21, 即每隔一个月：在论文图表中表示为u
                    # frame_idx = [i, i + 21, ... , i + 21*9]
                    # input_frames = [sequence[i], sequence[i + 21], ... , sequence[i + 21*9]]
                # Get w/o padding output frames


                output_frames = []
                output_frames_start_idx = i + input_span + rebalance - 1  # i + 190 + 21 -1 = i + 210
                for j in range(output_length):
                    output_frames.append(sequences[output_frames_start_idx + j * output_gap])
                data.append([frame_idx, input_frames, output_frames])

        # Log Sparse Self Attention
        elif sparse_type == 1:
            input_span = pow(2, input_length - 1)
            num_data = shape[0] - input_span - rebalance + 1
            print("input_span ", end='')
            print(input_span)
            for i in range(num_data):
                input_frames = []
                frame_idx = []
                target_index = i + input_span # 0 + 16
                for j in range(input_length):  # input_length = 5
                    frame_idx.append(target_index - pow(2, j))
                    # print(np.ceil((target_index - pow(2, j - 1))))
                    # exit()
                    input_frames.append(sequences[target_index - pow(2, j)])
                frame_idx.reverse()
                input_frames.reverse()
                output_frames = []
                output_frames_start_idx = i + input_span + rebalance - 1
                for j in range(output_length):
                    output_frames.append(sequences[output_frames_start_idx + j * output_gap])
                data.append([frame_idx, input_frames, output_frames])

        # LocalAttention + LogSparse Self Attention
        elif sparse_type == 2:
            input_span = pow(2, input_length - 1) + local_size #
            num_data = shape[0] - input_span - rebalance + 1
            print("input_span ", end='')
            print(input_span)
            for i in range(num_data):
                input_frames = []
                frame_idx = []
                target_index = i + input_span # 0 + 16

                for j in range(input_length + local_size):  # input_length = 4 local_size = 4
                    if j < local_size:
                        frame_idx.append(target_index - j)
                        input_frames.append(sequences[target_index - j])
                    else:
                        frame_idx.append(target_index - local_size - pow(2, (j - local_size)))
                        input_frames.append(sequences[target_index - local_size - pow(2, (j - local_size))])
                frame_idx.reverse()
                input_frames.reverse()
                output_frames = []
                output_frames_start_idx = i + input_span + rebalance - 1
                for j in range(output_length):
                    output_frames.append(sequences[output_frames_start_idx + j * output_gap])
                data.append([frame_idx, input_frames, output_frames])

        elif sparse_type == 3:
            print("updated")
            input_span = (pow(2, input_length - 1)-1) * lag_size + 1 # input_length = 3
            num_data = shape[0] - input_span - rebalance + 1
            print("input_span ", end='')
            print(input_span)
            for i in range(num_data):
                input_frames = []
                frame_idx = []
                target_index = i + input_span  # (0 + 2**(3-1) - 1) * 5 + 1
                for j in range(input_length):  # input_length = 5
                    frame_idx.append(target_index - ((pow(2, j)-1) * lag_size + 1))
                    input_frames.append(sequences[target_index - ((pow(2, j)-1) * lag_size + 1)])

                #print(target_index)
                frame_idx.reverse()
                input_frames.reverse()
                #print(len(input_frames))
                #print(frame_idx)

                output_frames = []
                output_frames_start_idx = i + input_span + rebalance - 1
                for j in range(output_length):
                    output_frames.append(sequences[output_frames_start_idx + j * output_gap])
                data.append([frame_idx, input_frames, output_frames])

        else:
            print("not implemented")

        # print(data)
        # print(data.shape)
        data = np.array(data)
        train_split_index = int(len(data) * split[0])
        validate_split_index = train_split_index + int(len(data) * split[1])
        train_data = data[:train_split_index]
        validate_data = data[train_split_index:validate_split_index]
        test_data = data[validate_split_index:]
        return train_data, validate_data, test_data

    d_covs = get_frame(covs, input_length, input_gap, output_length, output_gap, rebalance, split, sparse_type)
    d_cors = get_frame(cors, input_length, input_gap, output_length, output_gap, rebalance, split, sparse_type)

    return d_covs, d_cors
'''
for sparse type 0:
best parameter: 
    lag_t = 42
    input_gap = 21
    rebalance = 21
    input_length = 10
    output_length = 1
for sparse type 1:
best parameter: 
    lag_t = 21
    rebalance = 21
    input_length = 10
    output_length = 1
    lag_size = 1
for sparse type 2:
best parameter: 
    lag_t = 21
    rebalance = 21
    input_length = 10
    output_length = 1
    lag_size = 1
for sparse type 2:
best parameter: 
    lag_t = 21
    rebalance = 21
    input_length = 7
    output_length = 1
    lag_size = 5
'''
if __name__ == "__main__":
    closes = np.load("/home/xumingshi/Minimize_Portfolio_Risk/transformer/data/close_price.npy")
    closes = np.log(closes)[1:] - np.log(closes)[:-1]
    num_asset = 32
    lag_t = 21
    input_gap = 10
    rebalance = 21
    input_length = 10
    output_length = 1
    normal = True
    returns = closes[:, :32]
    pick = np.arange(32)
    cov_scale = "not_log"
    sparse_type = 2
    local_size = 5  # only for LocalAttention + LogSparse Self Attention
    lag_size = 5
    save_dir = '/home/xumingshi/Minimize_Portfolio_Risk/transformer/data/' + "num_%i_lag_%i_type_%i/" % (num_asset, lag_t, sparse_type)

    '''
    sparse_type:
    0: Linear Sparse Self Attention
    1: Log Sparse Self Attention
    2: LocalAttention + LogSparse Self Attention
    3: Log lagged Sparse Self Attention
    '''
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    np.save(save_dir + "pick.npy", pick)
    result = final_preprocess(returns, num_asset, lag_t, input_length=input_length, input_gap=input_gap, \
                              rebalance=rebalance, output_length=output_length, normal=normal, low=0, upper=1, \
                              cov_scale=cov_scale, sparse_type=sparse_type, local_size = local_size, lag_size = lag_size)
    np.save(save_dir + "result_gap_%i_horizon_%i.npy" % (input_gap, rebalance), result)