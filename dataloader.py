import time
from haversine import haversine_vector, Unit
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from joblib import Memory

#mem = Memory('/tmp/AIS_cahce')


#@mem.cache
def _initData(csv, resample_freq):
    data = pd.read_csv(csv, parse_dates=[0], infer_datetime_format=True)
    data['# Timestamp'] = pd.to_datetime(data['# Timestamp'])
    data = data.set_index(data['# Timestamp'])
    data = data.drop(columns=['Type of mobile'])

    # Unwrap all the angles to [0, inf[
    unwrap = lambda x: np.unwrap(np.deg2rad(x))
    data['COG'] = (data['COG'] + 1) * np.pi
    data['Heading'] = (data['Heading'] + 1) * np.pi

    data['COG'] = unwrap(data['COG'])
    data['Heading'] = unwrap(data['Heading'])

    # Resample to every 2 seconds and interpolate linearly between points
    data = data.groupby(['MMSI', pd.Grouper(freq=resample_freq)]).mean().reset_index()

    data = data.interpolate(method='linear')

    # Convert angles back to [-1, 1]
    data[['COG', 'Heading']] %= 2 * np.pi
    data['COG'] = data['COG'] / np.pi - 1
    data['Heading'] = data['Heading'] / np.pi - 1
    data = data.sort_values(['# Timestamp', 'MMSI']).reset_index().drop(columns='index')
    timestamps = data['# Timestamp'].unique()
    return data, timestamps


class SLDataset(Dataset):
    def __init__(self, csv_file, seq_len=20, resample_freq='20s'):
        self.seq_len = seq_len
        self.data, self.timestamps = _initData(csv_file, resample_freq)

    def __len__(self):
        return len(self.timestamps) - self.seq_len

    def __getitem__(self, idx):
        batch = []
        for step in self.timestamps[idx: idx + self.seq_len]:
            temp = self.data.loc[self.data['# Timestamp'] == step]
            temp = temp.drop(columns=['Width', 'Length', 'Draught', 'SOG', 'COG', 'ROT', 'Navigational status', '# Timestamp'])
            batch.append(temp.values.tolist())
        return batch


def sl_collate(batch):
    mmsi_all = []
    mmsi = []
    batch = batch[0]
    for idx, i in enumerate(batch):
        mmsi.append([])
        for j in i:
            mmsi[idx].append(j[0])
            if j[0] not in mmsi_all:
                mmsi_all.append(j[0])

    for idx, l in enumerate(mmsi):
        diff = list(set(mmsi_all) - set(l))

        for d in diff:
            batch[idx].append([d, -1, -1, -1])

    for b in batch:
        b.sort()

    return batch


def make_relative_meters(batch):
    end = []
    rel_x = []
    rel_y = []

    # find missing values:

    # make sure that last value is not -1
    batch = np.array(batch)

    for i in range(batch.shape[1]):
        if batch[-1, i, 1] == -1:

            for j in reversed(range(batch.shape[0])):
                if batch[j, i, 1] != -1:
                    j1 = j+1
                    if j1 == 0:
                        batch = np.delete(batch,i,1)
                    else:
                        for j in range(j1,batch.shape[0]):
                            if batch[j-2, i, 1] != -1:
                                batch[j, i, 1:3] = batch[j-1, i, 1:3] + (batch[j-1, i, 1:3] - batch[j-2, i, 1:3])
                            else:
                                batch[j, i, 1:3] = batch[j - 1, i, 1:3]
                        break

            # for j in reversed(range(batch.shape[0])):
            #     if batch[j, i, 1] != -1:
            #         batch[j:, i, :] = batch[j, i, :]
            #         break

    # find remainding values
    mask = np.where(batch[:, :, 1] == -1)

    # set to next value
    for i, j in zip(reversed(mask[0]), reversed(mask[1])):
        if i+2 < batch.shape[0]:
            batch[i, j, 1:3] = batch[i + 1, j, 1:3] + (batch[i+1, j, 1:3] - batch[i+2, j, 1:3])
        else:
            batch[i, j, 1:3] = batch[i + 1, j, 1:3]
        # batch[i, j, :] = batch[i + 1, j, :]

    # get last entry for each element in batch
    for l in batch[-1]:
        end.append((l[1], l[2]))

    for step in batch:
        step_x = []
        step_y = []
        rev = np.ones((2, batch.shape[1]))

        for idx, l in enumerate(step):
            step_x.append((l[1], end[idx][1]))
            step_y.append((end[idx][0], l[2]))

            if l[1] < end[idx][0]:
                rev[0, idx] *= -1

            if l[2] < end[idx][1]:
                rev[1, idx] *= -1

        rel_x.append(haversine_vector(step_x, end) * rev[0, :])
        rel_y.append(haversine_vector(step_y, end) * rev[1, :])

    return np.array(rel_x), np.array(rel_y)


def make_relative_meters_batch(batch, n_max=5):
    end = []
    rel_x_batch = []
    rel_y_batch = []
    batch = np.array(batch)
    out_x = np.zeros((batch.shape[1], batch.shape[0], n_max))
    out_y = np.zeros((batch.shape[1], batch.shape[0], n_max))

    # make sure that last value is not -1

    for i in range(batch.shape[1]):
        if batch[-1, i, 1] == -1:
            for j in reversed(range(batch.shape[0])):
                if batch[j, i, 1] != -1:
                    j1 = j+1
                    if j1 == 0:
                        batch = np.delete(batch,i,1)
                    else:
                        for j in range(j1,batch.shape[0]):
                            if batch[j-2, i, 1] != -1:
                                batch[j, i, 1:3] = batch[j-1, i, 1:3] + (batch[j-1, i, 1:3] - batch[j-2, i, 1:3])
                            else:
                                batch[j, i, 1:3] = batch[j - 1, i, 1:3]
                        break
            # for j in reversed(range(batch.shape[0])):
            #     if batch[j, i, 1] != -1:
            #         batch[j:, i, :] = batch[j, i, :]
            #         break

    # find remainding values
    mask = np.where(batch[:, :, 1] == -1)

    # set to next value
    for i, j in zip(reversed(mask[0]), reversed(mask[1])):
        if i+2 < batch.shape[0]:
            batch[i, j, 1:3] = batch[i + 1, j, 1:3] + (batch[i+1, j, 1:3] - batch[i+2, j, 1:3])
        else:
            batch[i, j, 1:3] = batch[i + 1, j, 1:3]

        # batch[i, j, :] = batch[i + 1, j, :]

    # get last entry for each element in batch
    for l in batch[-1]:
        end.append((l[1], l[2]))

    for i in range(batch.shape[1]):
        rel_x = []
        rel_y = []
        for step in batch:
            step_x = []
            step_y = []
            rev = np.ones((2, batch.shape[1]))

            for idx, l in enumerate(step):
                step_x.append((l[1], step[i][2]))
                step_y.append((step[i][1], l[2]))

                if l[1] < step[i][1]:
                    rev[0, idx] *= -1

                if l[2] < step[i][2]:
                    rev[1, idx] *= -1

            rel_x.append(haversine_vector(step_x, [tuple(step[i][1:3])] * len(step_x)) * rev[0, :])
            rel_y.append(haversine_vector(step_y, [tuple(step[i][1:3])] * len(step_x)) * rev[1, :])
        rel_x_batch.append(rel_x)
        rel_y_batch.append(rel_y)

    rel_x_batch = np.array(rel_x_batch)
    rel_y_batch = np.array(rel_y_batch)
    dist = np.sqrt(rel_x_batch ** 2 + rel_y_batch ** 2)
    sort_idx = np.argsort(dist[:, -1])
    sort_idx = np.delete(sort_idx, 0, 1)  # remove 0 distance

    for i in range(sort_idx.shape[0]):
        if sort_idx.shape[1] > n_max:
            idx_0 = rel_x_batch[i, :, sort_idx[i, :n_max]].T != 0
            out_x[i][idx_0] = rel_x_batch[i, :, sort_idx[i, :n_max]].T[idx_0]

            idx_0 = rel_y_batch[i, :, sort_idx[i, :n_max]].T != 0
            out_y[i][idx_0] = rel_y_batch[i, :, sort_idx[i, :n_max]].T[idx_0]
        else:
            idx_0 = rel_x_batch[i, :, sort_idx[i]].T != 0
            out_x[i, :, :sort_idx.shape[1]][idx_0] =  rel_x_batch[i, :, sort_idx[i]].T[idx_0]

            idx_0 = rel_y_batch[i, :, sort_idx[i]].T != 0
            out_y[i, :, :sort_idx.shape[1]][idx_0] =  rel_y_batch[i, :, sort_idx[i]].T[idx_0]

    return out_x, out_y


if __name__ == '__main__':
    dataset = SLDataset(csv_file='sep2018.csv', seq_len=40, resample_freq='20s')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=sl_collate)

    for d in dataloader:
        out = make_relative_meters_batch(d)
        print()
