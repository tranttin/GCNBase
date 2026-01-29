"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Shuxian Bi (stanbi@mail.ustc.edu.cn),Jianbai Ye (gusye@mail.ustc.edu.cn)
Design Dataset here
Every dataset's index has to start at 0
"""
import os
from os.path import join
import sys
import torch
import numpy as np
from scipy.sparse import csr_matrix, dok_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time
import datetime
from torch.utils.data import Dataset # Cần cho BasicDataset

class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")

    @property
    def n_users(self):
        raise NotImplementedError

    @property
    def m_items(self):
        raise NotImplementedError

    @property
    def trainDataSize(self):
        raise NotImplementedError

    @property
    def testDict(self):
        raise NotImplementedError

    @property
    def allPos(self):
        raise NotImplementedError

    def getUserItemFeedback(self, users, items):
        raise NotImplementedError

    def getUserPosItems(self, users):
        raise NotImplementedError

    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError

    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A =
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

    def get3DInteractionTensor(self):
        """
        Returns the Sparse User x Item x Time interaction tensor (NEW FUNCTION)
        """
        raise NotImplementedError

    # 🆕 Thêm các hàm cần thiết cho UITGCN
    def getSparseGraph_UT(self):
        raise NotImplementedError

    def getSparseGraph_IT(self):
        raise NotImplementedError

class LastFM(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    LastFM dataset
    """
    def __init__(self, path="../data/lastfm"):
        # train or test
        cprint("loading [last fm]")
        self.mode_dict = {'train':0, "test":1}
        self.mode    = self.mode_dict['train']
        # self.n_users = 1892
        # self.m_items = 4489
        trainData = pd.read_table(join(path, 'data1.txt'), header=None)
        # print(trainData.head())
        testData  = pd.read_table(join(path, 'test1.txt'), header=None)
        # print(testData.head())
        trustNet  = pd.read_table(join(path, 'trustnetwork.txt'), header=None).to_numpy()
        # print(trustNet[:5])
        trustNet -= 1
        trainData-= 1
        testData -= 1
        self.trustNet  = trustNet
        self.trainData = trainData
        self.testData  = testData
        self.trainUser = np.array(trainData[:][0])
        self.trainUniqueUsers = np.unique(self.trainUser)
        self.trainItem = np.array(trainData[:][1])
        # self.trainDataSize = len(self.trainUser)
        self.testUser  = np.array(testData[:][0])
        self.testUniqueUsers = np.unique(self.testUser)
        self.testItem  = np.array(testData[:][1])
        self.Graph = None
        print(f"LastFm Sparsity : {(len(self.trainUser) + len(self.testUser))/self.n_users/self.m_items}")

        # (users,users)
        self.socialNet    = csr_matrix((np.ones(len(trustNet)), (trustNet[:,0], trustNet[:,1]) ), shape=(self.n_users,self.n_users))
        # (users,items), bipartite graph
        self.UserItemNet  = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem) ), shape=(self.n_users,self.m_items))

        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_users)))
        self.allNeg = []
        allItems    = set(range(self.m_items))
        for i in range(self.n_users):
            pos = set(self._allPos[i])
            neg = allItems - pos
            self.allNeg.append(np.array(list(neg)))
        self.__testDict = self.__build_test()

    @property
    def n_users(self):
        return 1892

    @property
    def m_items(self):
        return 4489

    @property
    def trainDataSize(self):
        return len(self.trainUser)

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def getSparseGraph(self):
        if self.Graph is None:
            user_dim = torch.LongTensor(self.trainUser)
            item_dim = torch.LongTensor(self.trainItem)

            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            second_sub = torch.stack([item_dim+self.n_users, user_dim])
            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).int()
            self.Graph = torch.sparse.IntTensor(index, data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            dense = self.Graph.to_dense()
            D = torch.sum(dense, dim=1).float()
            D[D==0.] = 1.
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
            dense = dense/D_sqrt
            dense = dense/D_sqrt.t()
            index = dense.nonzero()
            data  = dense[dense >= 1e-9]
            assert len(index) == len(data)
            self.Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            self.Graph = self.Graph.coalesce().to(world.device)
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1, ))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def getUserNegItems(self, users):
        negItems = []
        for user in users:
            negItems.append(self.allNeg[user])
        return negItems



    def __getitem__(self, index):
        user = self.trainUniqueUsers[index]
        # return user_id and the positive items of the user
        return user

    def switch2test(self):
        """
        change dataset mode to offer test data to dataloader
        """
        self.mode = self.mode_dict['test']

    def __len__(self):
        return len(self.trainUniqueUsers)

class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    Dataset (gowalla, bkk, etc.)
    """

    def __init__(self,config = world.config,path="../data/gowalla"):
        # train or test
        cprint(f'loading [{path}]')
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        self.path = path
        self._3DInteractionTensor = None

        # --- CẤU HÌNH TEMPORAL: DÙNG LẠI 168 SLOTS ---
        self.n_time_slots = 168
        self.min_time = None
        self.time_slot_seconds = 3600 # 1 giờ
        # -----------------------------------------------

        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0

        # --- Tải train.txt và test.txt ---
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    if len(l) == 1: continue
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items) if items else 0)
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    if len(l) == 1: continue
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items) if items else 0)
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)
        self.m_item += 1
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)

        # --- Tải và xử lý dữ liệu thời gian (168 slots) ---
        self.trainTimeSlot = self.load_temporal_data(path, is_train=True)
        self.testTimeSlot = self.load_temporal_data(path, is_train=False)

        # ❗ KIỂM TRA LẠI KÍCH THƯỚC:
        if len(self.trainUser) != len(self.trainTimeSlot):
             cprint(f"FATAL ERROR: Train Interaction ({len(self.trainUser)}) and Time ({len(self.trainTimeSlot)}) arrays mismatch.", "red")
             sys.exit(1)


        self.Graph = None # UI Graph
        self.Graph_UT = None # User-Time Graph
        self.Graph_IT = None # Item-Time Graph

        cprint(f"{self.trainDataSize} interactions for training")
        cprint(f"{self.testDataSize} interactions for testing")
        cprint(f"Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))

        # 🆕 Xây dựng UT và IT NETS (Interaction Matrix)
        # User-Time Interaction Matrix (U x T)
        self.UserTimeNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainTimeSlot)),
                                      shape=(self.n_user, self.n_time_slots))
        # Item-Time Interaction Matrix (I x T)
        self.ItemTimeNet = csr_matrix((np.ones(len(self.trainItem)), (self.trainItem, self.trainTimeSlot)),
                                      shape=(self.m_item, self.n_time_slots))


        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        cprint(f"Dataset is ready to go. Time slots: {self.n_time_slots}")

    # --- HÀM MỚI: Tải và Xử lý Dữ liệu Thời gian (Sử dụng 168 slots) ---
    def load_temporal_data(self, path, is_train=True):
        """
        Tải và chuyển đổi Unix Timestamp sang Time Slot Index (0 -> n_time_slots - 1)
        """
        file_name = 'time_train.txt' if is_train else 'time_test.txt'
        time_file = path + '/' + file_name

        time_data = []
        try:
            with open(time_file) as f:
                for l in f.readlines():
                    if len(l) > 0:
                        l = l.strip('\n').split(' ')
                        if len(l) == 1: continue
                        times = [int(i) for i in l[1:]]
                        time_data.extend(times)
        except FileNotFoundError:
             cprint(f"WARNING: Time file {file_name} not found. Temporal component will be skipped.", "yellow")
             return np.array([], dtype=int)


        time_array = np.array(time_data)

        if is_train:
            if len(time_array) == 0:
                self.min_time = 0
            else:
                self.min_time = time_array.min()
            cprint(f"Min Unix Time (used for normalization): {self.min_time}")

        if self.min_time is None:
            raise ValueError("Error: self.min_time must be set by the training data.")

        if len(time_array) == 0:
            time_slot_indices = np.array([], dtype=int)
        else:
            # Chuyển đổi Unix Time sang Time Slot Index (0 -> 167)
            time_slot_indices = (time_array - self.min_time) // self.time_slot_seconds
            time_slot_indices %= self.n_time_slots

        return time_slot_indices

    # --- HÀM MỚI: Tensor Tương tác 3D (Sparse) ---
    def get3DInteractionTensor(self):
        """
        Builds and returns the Sparse User x Item x Time interaction tensor.
        """
        if self._3DInteractionTensor is not None:
            return self._3DInteractionTensor

        if len(self.trainUser) == 0 or len(self.trainItem) == 0 or len(self.trainTimeSlot) == 0:
            cprint("WARNING: Train data is empty or temporal data is missing. Cannot generate 3D Interaction Tensor.", "yellow")
            return torch.sparse_coo_tensor(
                torch.empty((3, 0), dtype=torch.long),
                torch.empty(0, dtype=torch.float),
                size=(self.n_user, self.m_item, self.n_time_slots),
                device=world.device
            )

        cprint("Generating 3D Interaction Tensor (Sparse)...")

        # Tọa độ (User, Item, Time Slot)
        coords = np.stack([self.trainUser, self.trainItem, self.trainTimeSlot], axis=1)
        values = np.ones(len(self.trainUser))

        coords_torch = torch.LongTensor(coords).t()
        values_torch = torch.FloatTensor(values)

        interaction_tensor_3d = torch.sparse_coo_tensor(
            coords_torch,
            values_torch,
            size=(self.n_user, self.m_item, self.n_time_slots),
            device=world.device
        )

        self._3DInteractionTensor = interaction_tensor_3d.coalesce()
        cprint("3D Interaction Tensor generated.")
        return self._3DInteractionTensor

    # --- HÀM MỚI: Ma trận Kề cho UT Graph ---
    def getSparseGraph_UT(self):
        """Builds and returns the normalized Adjacency Matrix for User-Time Graph (U x T)."""
        if self.Graph_UT is None:
            # Tên file cho UT Graph
            ut_path = self.path + '/s_pre_adj_mat_UT.npz'
            try:
                pre_adj_mat = sp.load_npz(ut_path)
                cprint("successfully loaded UT Graph...", "green")
                norm_adj = pre_adj_mat
            except FileNotFoundError:
                cprint("generating UT adjacency matrix...", "green")
                s = time()

                # 1. Tạo ma trận kề tổng thể (U+T) x (U+T)
                adj_mat = dok_matrix((self.n_user + self.n_time_slots, self.n_user + self.n_time_slots), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R_UT = self.UserTimeNet.tolil()

                # 2. Gán R_UT và R_UT.T
                adj_mat[:self.n_user, self.n_user:] = R_UT
                adj_mat[self.n_user:, :self.n_user] = R_UT.T
                adj_mat = adj_mat.todok()

                # 3. Chuẩn hóa (Normalization)
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()

                cprint(f"costing {time()-s}s, saved UT norm_mat...", "green")
                sp.save_npz(ut_path, norm_adj)

            self.Graph_UT = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph_UT = self.Graph_UT.coalesce().to(world.device)

        return self.Graph_UT

    # --- HÀM MỚI: Ma trận Kề cho IT Graph ---
    def getSparseGraph_IT(self):
        """Builds and returns the normalized Adjacency Matrix for Item-Time Graph (I x T)."""
        if self.Graph_IT is None:
            # Tên file cho IT Graph
            it_path = self.path + '/s_pre_adj_mat_IT.npz'
            try:
                pre_adj_mat = sp.load_npz(it_path)
                cprint("successfully loaded IT Graph...", "green")
                norm_adj = pre_adj_mat
            except FileNotFoundError:
                cprint("generating IT adjacency matrix...", "green")
                s = time()

                # 1. Tạo ma trận kề tổng thể (I+T) x (I+T)
                adj_mat = dok_matrix((self.m_item + self.n_time_slots, self.m_item + self.n_time_slots), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R_IT = self.ItemTimeNet.tolil()

                # 2. Gán R_IT và R_IT.T
                adj_mat[:self.m_item, self.m_item:] = R_IT
                adj_mat[self.m_item:, :self.m_item] = R_IT.T
                adj_mat = adj_mat.todok()

                # 3. Chuẩn hóa (Normalization)
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()

                cprint(f"costing {time()-s}s, saved IT norm_mat...", "green")
                sp.save_npz(it_path, norm_adj)

            self.Graph_IT = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph_IT = self.Graph_IT.coalesce().to(world.device)

        return self.Graph_IT


    # --- Các hàm LightGCN gốc (giữ nguyên) ---

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n_user + self.m_item) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_user + self.m_item
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                cprint("successfully loaded UI Graph...", "green")
                norm_adj = pre_adj_mat
            except FileNotFoundError :
                cprint("generating UI adjacency matrix...", "green")
                s = time()
                adj_mat = dok_matrix((self.n_user + self.m_item, self.n_user + self.m_item), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()

                # FIX LỖI SHAPE MISMATCH:
                adj_mat[:self.n_user, self.n_user:] = R
                adj_mat[self.n_user:, :self.n_user] = R.T

                adj_mat = adj_mat.todok()

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                cprint(f"costing {end-s}s, saved UI norm_mat...", "green")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("don't split the matrix")
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    # def getUserNegItems(self, users):
    #     negItems = []
    #     for user in users:
    #         negItems.append(self.allNeg[user])
    #     return negItems

    def __getitem__(self, index):
        user = self.trainUniqueUsers[index]
        # return user_id and the positive items of the user
        return user

    def switch2test(self):
        """
        change dataset mode to offer test data to dataloader
        """
        self.mode = self.mode_dict['test']

    def __len__(self):
        return len(self.trainUniqueUsers)