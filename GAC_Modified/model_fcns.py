from torch import nn
from torch.nn import functional as F
import open3d as o3d
import numpy as np
from sklearn.neighbors import KDTree
import torch
import copy
from Config import Hparams as Config


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sampler(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def feat_gen(data):

    # print("Finding nearest neighbors...")
    kdt = KDTree(data, leaf_size=30, metric='euclidean')
    index = kdt.query(data, k=30)
    qpnt_nbrs = [i for i in index[1]]
    # # qpnt_nbrs = [fullset[i] for i in index]  # each query point's neighbors
    # verticality = []
    # e_entropy = []
    saliency1 = []
    saliency2 = []
    saliency3 = []
    for nbrs in qpnt_nbrs:
        spatial_nbrs = data[nbrs]
        spatial_nbrs = spatial_nbrs[:, :3]

            # covariance matrix, eigen value and vector calculation
        cov_mat = np.cov(spatial_nbrs.T)
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)
        norm_eig_vals = [number / sum(eig_vals) for number in eig_vals]
        # The eig_vec of the smallest eig_val is the normal. hence eigen pairs are formed in order to accurately
        # retrieve the normals
        eig_pairs = [(np.abs(norm_eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
        eig_pairs.sort()
        eig_pairs.reverse()
        # # Verticality (variation)
        # e1 = eig_pairs[0][1]
        # e2 = eig_pairs[1][1]
        # e3 = eig_pairs[2][1]
        # unary_vec = [eig_pairs[0][0] * fabs(e1[0]) + eig_pairs[1][0] * fabs(e2[0]) + eig_pairs[2][0] * fabs(e3[0]),
        #              eig_pairs[0][0] * fabs(e1[1]) + eig_pairs[1][0] * fabs(e2[1]) + eig_pairs[2][0] * fabs(e3[1]),
        #              eig_pairs[0][0] * fabs(e1[2]) + eig_pairs[1][0] * fabs(e2[2]) + eig_pairs[2][0] * fabs(e3[2])]
        # normal = np.sqrt(unary_vec[0] * unary_vec[0] + unary_vec[1] * unary_vec[1] + unary_vec[2] * unary_vec[2])
        # g4 = unary_vec[2] / normal
        # verticality.append(g4)

        # # calculating eigen-entropy (EE)
        # g5 = -eig_pairs[0][0]*np.log(eig_pairs[0][0]) - eig_pairs[1][0]*np.log(eig_pairs[1][0]) - eig_pairs[2][0]*np.log(eig_pairs[2][0])
        # e_entropy.append(g5)

        # calculating saliency features (L, P, S)
        g1 = (eig_pairs[0][0] - eig_pairs[1][0]) / eig_pairs[0][0]
        saliency1.append(g1)
        g2 = (eig_pairs[1][0] - eig_pairs[2][0]) / eig_pairs[0][0]
        saliency2.append(g2)
        g3 = eig_pairs[2][0] / eig_pairs[0][0]
        saliency3.append(g3)
    return np.column_stack([saliency1, saliency2, saliency3])


def struct_aware_sampler(xyz, npoint, radius=0.04, mode='du_ud_full'):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
        mode = 'du_ud_full' default     # du_full, normal_plane, du_ud_full
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    np_xyz = xyz.to('cpu').numpy()
    B, N, C = np_xyz.shape
    # centroids = np.zeros((B, npoint, C))
    centroids = np.zeros((B, npoint))
    for batch in range(B):
        # From numpy to Open3D
        xyz_pcd = o3d.geometry.PointCloud()
        xyz_pcd.points = o3d.utility.Vector3dVector(np_xyz[batch, :, :])
        xyz_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

        lps = feat_gen(data=np_xyz[batch, :, :])
        pcd_tree = o3d.geometry.KDTreeFlann(xyz_pcd)
         # rand_idx_for_try = [1500]
        deltas = []
        for i in range(len(lps)):
            [k, idx, _] = pcd_tree.search_hybrid_vector_3d(xyz_pcd.points[i], radius=radius, max_nn=30)
            diff_dist = np.mean(np.sum((np.asarray(xyz_pcd.normals)[idx, :] - np.asarray(xyz_pcd.normals)[i, :]) ** 2, -1))
            lps_dists = np.mean(np.abs(lps[idx, :] - lps[i, :]), axis=0)
            temp = np.column_stack([diff_dist, *lps_dists])
            deltas.append(temp)

        if mode == 'du_full':  # neat, but lacks a little in places of uniformity
            deltas = np.concatenate(deltas, axis=0)
            feat_sum = np.sum(deltas, axis=-1)
            sort_idx = np.argsort(feat_sum)
            sorted_xyz = np_xyz[batch, sort_idx, :]
            samples = sorted_xyz[-npoint:, :]  # high mean values
        elif mode == 'du_ud_full':  # good
            # nsamples = len(np_xyz)//8
            deltas = np.concatenate(deltas, axis=0)
            feat_sum = np.sum(deltas, axis=-1)
            sort_idx = np.argsort(feat_sum)
            du_idx = sort_idx[-(int(np.floor(npoint * 0.7))):]
            # sorted_xyz = np_xyz[batch, sort_idx, :]
            # du = sorted_xyz[-(int(np.floor(npoint * 0.7))):, :]  # high mean values
            rand_idx = np.random.choice(int(N - int(np.floor(npoint * 0.7))),
                                        int(np.ceil(npoint * 0.3)), replace=False)
            # rand = sorted_xyz[rand_idx, :]
            samples_idx = np.concatenate([du_idx, rand_idx], axis=0)
        elif mode == 'normal_plane':  # or normal_only. but both r not too good
            # nsamples = len(np_xyz) // 8
            deltas = np.concatenate(deltas, axis=0)
            plane_sort_idx = np.argsort(deltas[:, 1])
            plane_sorted_xyz = np_xyz[plane_sort_idx, :]
            normal_sort_idx = np.argsort(deltas[:, 0])
            normal_sorted_xyz = np_xyz[normal_sort_idx, :]
            samples = np.concatenate([normal_sorted_xyz[-(int(np.floor(npoint * 0.7))):, :],  # high variance
                                      plane_sorted_xyz[:(int(np.floor(npoint * 0.3))), :]], axis=0)  # low variance
        centroids[batch, :] = samples_idx
    centroids = torch.from_numpy(centroids).long().to(device)
    return centroids


def query_ball_point(radius, xyz, new_xyz, nscale):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)

    # toggling btwn gpu n cpu n back
    device = "cpu"
    torch.device(device)
    group_idx = group_idx.to(device)
    sqrdists = sqrdists.to(device)

    group_idx[sqrdists > radius ** 2] = N
    if nscale == 2:
        group_idx0 = group_idx.sort(dim=-1)[0][:, :, :Config.multi_nsamples[0]]
        group_idx1 = group_idx.sort(dim=-1)[0][:, :, :Config.multi_nsamples[1]]
    else:
        group_idx0 = group_idx.sort(dim=-1)[0][:, :, :Config.single_nsample]

    device = "cuda:0"
    torch.device(device)
    
    if nscale == 2:
        group_idx0 = group_idx0.to(device)
        group_first0 = group_idx0[:, :, 0].view(B, S, 1).repeat([1, 1, Config.multi_nsamples[0]])
        mask = group_idx0 == N
        group_idx0[mask] = group_first0[mask]

        group_idx1 = group_idx1.to(device)
        group_first1 = group_idx1[:, :, 0].view(B, S, 1).repeat([1, 1, Config.multi_nsamples[1]])
        mask = group_idx1 == N
        group_idx1[mask] = group_first1[mask]
    # sqrdists = sqrdists.to(device)
    else:
        group_idx0 = group_idx0.to(device)
        group_first0 = group_idx0[:, :, 0].view(B, S, 1).repeat([1, 1, Config.single_nsample])
        mask = group_idx0 == N
        group_idx0[mask] = group_first0[mask]
        group_idx1 = None
    
    return group_idx0, group_idx1


def sample_and_group(sampler, npoint, nscale, radius, xyz, points, returnfps=False):
    """
    Input:
        sampler: 'sas', 'fps'
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = npoint
    if sampler == 'fps':
        fps_idx = farthest_point_sampler(xyz, npoint)  # [B, npoint, C] sampled fps idxes
        new_xyz = index_points(xyz, fps_idx)  # sampled xyz based on fps_idx
    else:
        sas_idx = struct_aware_sampler(xyz, npoint, radius=1.6)
        new_xyz = index_points(xyz, sas_idx)  # sampled xyz based on fps_idx
    idx0, idx1 = query_ball_point(radius, xyz, new_xyz, nscale)

    if nscale == 2:
        grouped_xyz0 = index_points(xyz, idx0)  # [B, npoint, nsample, C]
        grouped_xyz1 = index_points(xyz, idx1)  # [B, npoint, nsample, C]
        grouped_xyz_norm0 = grouped_xyz0 - new_xyz.view(B, S, 1, C)
        grouped_xyz_norm1 = grouped_xyz1 - new_xyz.view(B, S, 1, C)
        if points is not None:
            grouped_points0 = index_points(points, idx0)
            grouped_points1 = index_points(points, idx1)
            if sampler == 'fps':
                sampled_points = index_points(points, fps_idx)
            else:
                sampled_points = index_points(points, sas_idx)
            sampled_points = torch.cat([new_xyz, sampled_points], dim=-1)
            new_points0 = torch.cat([grouped_xyz_norm0, grouped_points0], dim=-1)  # [B, npoint, nsample, C+D]
            new_points1 = torch.cat([grouped_xyz_norm1, grouped_points1], dim=-1)  # [B, npoint, nsample, C+D]
        else:
            new_points0 = grouped_xyz_norm0
            new_points1 = grouped_xyz_norm1
            sampled_points = new_xyz
        if returnfps:
            return new_xyz, new_points0, new_points1, grouped_xyz0, grouped_xyz1, sampled_points
        else:
            return new_xyz, new_points0, new_points1
    else:
        grouped_xyz0 = index_points(xyz, idx0)  # [B, npoint, nsample, C]
        grouped_xyz_norm0 = grouped_xyz0 - new_xyz.view(B, S, 1, C)
        grouped_xyz1 = None

        if points is not None:
            grouped_points0 = index_points(points, idx0)
            if sampler == 'fps':
                sampled_points = index_points(points, fps_idx)
            else:
                sampled_points = index_points(points, sas_idx)
            sampled_points = torch.cat([new_xyz, sampled_points], dim=-1)
            new_points0 = torch.cat([grouped_xyz_norm0, grouped_points0], dim=-1)  # [B, npoint, nsample, C+D]
            new_points1 = None
        else:
            new_points0 = grouped_xyz_norm0
            new_points1 = None
            sampled_points = new_xyz
        if returnfps:
            return new_xyz, new_points0, new_points1, grouped_xyz0, grouped_xyz1, sampled_points
        else:
            return new_xyz, new_points0, new_points1


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


def max_dilation(xyz, points, layer, sampler='fps'):
    """determine the max nbrhood"""
    """
    Input:
        xyz: input points position data, [B, C, N]
        points: input points data, [B, D, N]
    Return:
        new_(center)_xyz: sampled points position data, [B, C, S]
        new_(center)_points_concat: sample points feature data, [B, D', S]
        grouped_xyz: group xyz data [B, npoint, nsample, C]
        grouped_feature: sampled points feature [B, npoint, nsample, D]

    """
    xyz = xyz.permute(0, 2, 1)
    if points is not None:
        points = points.permute(0, 2, 1)
    # if nscales == 3:
    #     max_nsamples = Config.multi_nsamples[2]
    # elif nscales == 2:
    #     max_nsamples = Config.multi_nsamples[1]
    # elif nscales == 1:
    #     max_nsamples = Config.single_nsample
    new_xyz, new_points0, new_points1, grouped_xyz0, grouped_xyz1, fps_points = sample_and_group(sampler,
                                                                                    Config.npoint[layer],
                                                                                    Config.nscales[layer],
                                                                                    Config.max_radius[layer], 
                                                                                    xyz, points, True)
    return new_xyz, new_points0, new_points1, grouped_xyz0, grouped_xyz1, fps_points


def pna_support(attention, grp_feat):
    from pna.aggregators import AGGREGATORS
    from pna.scalers import SCALERS

    _,_,_,D = grp_feat.shape
    aggregators = Config.aggregators
    scalers = Config.scalers
    aggregators = [AGGREGATORS[aggr] for aggr in aggregators.split(' ')]
    scalers = [SCALERS[scale] for scale in scalers.split(' ')]

    # attention as alt to adj
    thresh_att = attention * (attention < Config.threshold)  # [B, npoint, nsample,D]
    print(thresh_att[1, 273, :], '\nmin:', torch.min(attention), '\nmax:', torch.max(attention))
    att_adj = thresh_att.where(thresh_att == 0.0, torch.Tensor([1.0]).to(thresh_att.device))
    print(torch.sum(att_adj, dim=2)[1, :])
    N_deg = torch.sum(att_adj, dim=-1)  # npoint node deg # [B N]
    # N_deg = torch.round(torch.mean(torch.sum(att_adj, dim=2), dim=-1))  # [B N]
    print(N_deg[1,:])
    avg_d = torch.mean(torch.log(N_deg + 1))  # scaler

    # aggregation
    """
    1st m output = [B N Din * num of aggregators employed]
    2nd m output = [B N Din * num of scalers employed]
    i.e., m output = [B N (Din * num of aggregators employed) * num of scalers employed]
    """
    m = torch.cat([aggregate(grp_feat, att_adj, N_deg) for aggregate in aggregators], dim=2)
    m = torch.cat([scale(m, N_deg, avg_d=avg_d) for scale in scalers], dim=2)
    return m


class GATLayer(nn.Module):
    def __init__(self, all_channel, feat_dim, dropout, alpha):
        super(GATLayer, self).__init__()
        self.alpha = alpha
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.a = nn.Parameter(torch.zeros(size=(all_channel, feat_dim)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, center_xyz, center_feature, grouped_xyz, grouped_feature):
        """
        Inputs:
            :param center_xyz: sampled points position data [B, npoint, C]
            :param center_feature: centered points' features [B, npoint, D]
            :param grouped_xyz: grouped xyz data [B, npoint, nsample, C]
            :param grouped_feature: sampled points' features [B, npoint, nsample, C]
        Return:
            pooled graph: results of graph pooling [B, npoint, D]
        """
        B, npoint, C = center_xyz.size()
        _, _, nsample, D = grouped_feature.size()
        delta_p = center_xyz.view(B, npoint, 1, C).expand(B, npoint, nsample,
                                                          C) - grouped_xyz  # [B, npoint, nsample, C]
        delta_h = center_feature.view(B, npoint, 1, D).expand(B, npoint, nsample,
                                                              D) - grouped_feature  # [B, npoint, nsample, D]
        delta_p_concat_h = torch.cat([delta_p, delta_h], dim=-1)  # [B, npoint, nsample, C+D]
        e = self.leakyrelu(torch.matmul(delta_p_concat_h, self.a))  # [B, npoint, nsample,D]
        attention = F.softmax(e, dim=2)  # [B, npoint, nsample,D]
        attention = F.dropout(attention, self.dropout, training=self.training)
        graph_pooling = torch.sum(torch.mul(attention, grouped_feature), dim=2)  # [B, npoint, D]
        return graph_pooling


class GACLayer(nn.Module):
    def __init__(self, npoint, radius, nscale, in_channel, mlp, group_all=False, dropout=0.5, alpha=0.2):
        super(GACLayer, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nscale = nscale
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.dropout = dropout
        self.alpha = alpha
        self.residual = None

        if nscale > 1:
            self.mlp_convs1 = nn.ModuleList()
            self.mlp_bns1 = nn.ModuleList()
            last_channel = in_channel  # the first last_channel == in_channel == 3 + 4 (pos + rgb,i)
            for out_channel in mlp:  # [32, 32, 64]
                self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
                self.mlp_bns.append(nn.BatchNorm2d(out_channel, track_running_stats=False))
                self.mlp_convs1.append(nn.Conv2d(last_channel, out_channel, 1))
                self.mlp_bns1.append(nn.BatchNorm2d(out_channel, track_running_stats=False))
                last_channel = out_channel
            self.GAT = AttentionMix(10+last_channel,last_channel,self.dropout,self.alpha)
            self.GAT1 = AttentionMix(10+last_channel,last_channel,self.dropout,self.alpha)
        else:
            last_channel = in_channel  # the first last_channel == in_channel == 3 + 4 (pos + rgb,i)
            for out_channel in mlp:  # [32, 32, 64]
                self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
                self.mlp_bns.append(nn.BatchNorm2d(out_channel, track_running_stats=False))
                last_channel = out_channel
            self.GAT = AttentionMix(10+last_channel,last_channel,self.dropout,self.alpha)

        self.group_all = group_all
        

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points0, new_points1, grouped_xyz0, grouped_xyz1, fps_points = sample_and_group('fps', self.npoint, self.nscale, self.radius, xyz, points, True)
        # db = {'a':new_xyz, 'b':new_points, 'c':grouped_xyz, 'd':fps_points}
        # torch.save(db, 's_n_g_output')
        # new_xyz: sampled (centre) points position (xyz) data, [B, npoint, C]
        # new_points: sampled points data (grouped), [B, npoint, nsample, C+D]
        # fps_points: [B, npoint, C+D,1]
        new_points0 = new_points0.permute(0, 3, 2, 1)  # [B, C+D, nsample, npoint]
        if new_points1 is not None:
            new_points1 = new_points1.permute(0, 3, 2, 1)  # [B, C+D, nsample, npoint]
        fps_points0 = fps_points.unsqueeze(3).permute(0, 2, 3, 1)  # [B, C+D, 1, npoint]
        self.residual = fps_points0.clone()
        fps_points1 = fps_points0.clone()
        if self.nscale == 2:
            for i, conv in enumerate(self.mlp_convs):
                bn = self.mlp_bns[i]
                fps_points0 = F.relu(bn(conv(fps_points0)))
                new_points0 = F.relu(bn(conv(new_points0)))   
            for i, conv in enumerate(self.mlp_convs1):
                bn = self.mlp_bns1[i]
                fps_points1 = F.relu(bn(conv(fps_points1)))
                new_points1 = F.relu(bn(conv(new_points1)))  
            
            new_points0 = self.GAT(center_xyz=new_xyz,
                                center_feature=fps_points0.squeeze().permute(0,2,1),
                                grouped_xyz=grouped_xyz0, 
                                grouped_feature=new_points0.permute(0,3,2,1))
            new_points1 = self.GAT1(center_xyz=new_xyz,
                                center_feature=fps_points1.squeeze().permute(0,2,1),
                                grouped_xyz=grouped_xyz1, 
                                grouped_feature=new_points1.permute(0,3,2,1))  

            new_points0 = (new_points0 + new_points1)/2 #TODO: newpoints0 = find the mean of newpoints 0 and 1                    
        else:
            for i, conv in enumerate(self.mlp_convs):
                bn = self.mlp_bns[i]
                fps_points0 = F.relu(bn(conv(fps_points0)))
                new_points0 = F.relu(bn(conv(new_points0)))
            # new_points: [B, F, nsample,npoint]
            # fps_points: [B, F, 1,npoint]
            new_points0 = self.GAT(center_xyz=new_xyz,
                                center_feature=fps_points0.squeeze().permute(0,2,1),
                                grouped_xyz=grouped_xyz0, 
                                grouped_feature=new_points0.permute(0,3,2,1))

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points0 = new_points0.permute(0, 2, 1)
        return new_xyz, new_points0


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel, track_running_stats=False))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            dists[dists < 1e-10] = 1e-10
            weight = 1.0 / dists  # [B, N, 3]
            weight = weight / torch.sum(weight, dim=-1).view(B, N, 1)  # [B, N, 3]
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        return new_points


class GACNet(nn.Module):
    # graph attention convolution layer
    def __init__(self, num_classes, dropout=0, alpha=0.2):
        super(GACNet, self).__init__()
        # convs & GAT
        # GraphAttentionConvLayer: npoint, radius, nsample, in_channel, mlp, group_all,dropout,alpha
        self.sa1 = GACLayer(1024, 0.2, 32, 3 + 3, [32, 32, 64], False, dropout, alpha)
        self.sa2 = GACLayer(256, 0.4, 32, 64 + 3, [64, 64, 128], False, dropout, alpha)
        self.sa3 = GACLayer(64, 0.6, 32, 128 + 3, [128, 128, 256], False, dropout, alpha)
        self.sa4 = GACLayer(16, 0.8, 32, 256 + 3, [256, 256, 512], False, dropout, alpha)

        #pointNet feature propagation: in_channel, mlp
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

        # FC layers
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128, track_running_stats=False)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(128, Config.num_classes, 1)

    def forward(self, xyz, point):
        l1_xyz, l1_points = self.sa1(xyz, point)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x


class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bn=False, act_fxn=False):
        super(SharedMLP, self).__init__()
        # shared mlp block
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.norm = bn
        self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99, track_running_stats=False) if bn else None
        self.act_fxn = act_fxn

    def forward(self, feats):
        # shared mlp block
        x = self.conv(feats)
        if self.norm:
            x = self.bn1(x)
        if self.act_fxn:
            x = nn.functional.relu(x)
        return x


class AttentionGAC(nn.Module):
    def __init__(self,all_channel,feature_dim,dropout,alpha):
        super(AttentionGAC, self).__init__()
        self.alpha = alpha
        self.a = nn.Parameter(torch.zeros(size=(all_channel, feature_dim)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, center_xyz, center_feature, grouped_xyz, grouped_feature):
        '''
        Input:
            center_xyz: sampled points position data [B, npoint, C]
            center_feature: centered point feature [B, npoint, D]
            grouped_xyz: group xyz data [B, npoint, nsample, C]
            grouped_feature: sampled points feature [B, npoint, nsample, D]
        Return:
            graph_pooling: results of graph pooling [B, npoint, D]
        '''
        B, npoint, C = center_xyz.size()
        _, _, nsample, D = grouped_feature.size()
        delta_p = center_xyz.view(B, npoint, 1, C).expand(B, npoint, nsample, C) - grouped_xyz # [B, npoint, nsample, C]
        delta_h = center_feature.view(B, npoint, 1, D).expand(B, npoint, nsample, D) - grouped_feature # [B, npoint, nsample, D]
        delta_p_concat_h = torch.cat([delta_p,delta_h], dim=-1) # [B, npoint, nsample, C+D]
        e = self.leakyrelu(torch.matmul(delta_p_concat_h, self.a)) # [B, npoint, nsample,D]
        attention = F.softmax(e, dim=2) # [B, npoint, nsample,D]
        attention = F.dropout(attention, self.dropout, training=self.training)
        graph_pooling = torch.sum(torch.mul(attention, grouped_feature), dim=2)  # [B, npoint, D]
        return graph_pooling


class AttentionMix(nn.Module):
    """more rel. point pos. features encoded (i.e. x_i, x_j, x_i - X_j, x_ij dist) 
       added to GACNet's attention computation"""
    def __init__(self,all_channel,feature_dim,dropout,alpha, pna=False):
        super(AttentionMix, self).__init__()
        self.pna = pna
        self.alpha = alpha
        self.a = nn.Parameter(torch.zeros(size=(feature_dim * Config.tot_aggs * Config.tot_scalers,
                                                feature_dim)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.mlp1 = SharedMLP(10, 10, bn=True, act_fxn=True)
        self.mlp2 = nn.Linear(in_features=all_channel, out_features=feature_dim)
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, center_xyz, center_feature, grouped_xyz, grouped_feature):
        '''
        Input:
            center_xyz: sampled points position data [B, npoint, C]
            center_feature: centered point feature [B, npoint, D]
            grouped_xyz: group xyz data [B, npoint, nsample, C]
            grouped_feature: sampled points feature [B, npoint, nsample, D]
        Return:
            graph_pooling: results of graph pooling [B, npoint, D]
        '''
        B, npoint, C = center_xyz.size()
        _, _, nsample, D = grouped_feature.size()
        delta_p = center_xyz.view(B, npoint, 1, C).expand(B, npoint, nsample, C) - grouped_xyz # [B, npoint, nsample, C]
        dist = torch.sqrt(torch.sum(delta_p ** 2, dim=-1)).unsqueeze(-1)
        delta_h = center_feature.view(B, npoint, 1, D).expand(B, npoint, nsample, D) - grouped_feature # [B, npoint, nsample, D]
        concat = torch.cat([center_xyz.view(B, npoint, 1, C).expand(B, npoint, nsample, C),
                            grouped_xyz, delta_p, dist], dim=-1)  # [B, npoint, nsample, C+7]
        # todo: chk both leakyrelu n relu activations
        sharedmlp = self.mlp1(concat.permute(0, 3, 2, 1))
        feat_concat = torch.cat([sharedmlp.permute(0, 3, 2, 1), delta_h], dim=-1)
        e = self.leakyrelu(self.mlp2(feat_concat))  # [B, npoint, nsample, D]

        if self.pna:  # todo: dropout or not, secof gate-like function of adj
            delta_p_concat_h = torch.cat([delta_p, delta_h], dim=-1)  # [B, npoint, nsample, C+D]
            dists = torch.sum(torch.square(delta_p_concat_h), dim=-1)
            softmax_dists = torch.softmax(dists, dim=-1)

            attention = F.softmax(e, dim=2)  # [B, npoint, nsample,D]
            # print(torch.sigmoid(softmax_att)[1, 432, :, :])
            # attention = F.dropout(attention, self.dropout, training=self.training)  # todo: drop-out here if needed
            att_grp_feat = torch.mul(attention, grouped_feature)  # attended features
            # sigmoid_att = 1 / (1 + torch.exp(e))  # [B, npoint, nsample,D]

            # aggrerating attended feats
            graph_pooling = pna_support(softmax_dists, att_grp_feat)  # [B, npoint, nsample, D']
            if Config.post_trans:
                # mlp of m to return D' to original Din
                graph_pooling = torch.matmul(graph_pooling, self.a)  # todo: generating nan, fix
        else:
            attention = F.softmax(e, dim=2)  # [B, npoint, nsample,D]
            attention = F.dropout(attention, self.dropout, training=self.training)
            graph_pooling = torch.sum(torch.mul(attention, grouped_feature), dim=2)  # [B, npoint, D]
        return graph_pooling


class GACLayer_NoSampling(nn.Module):
    """this GACLayer does not include the sampling step, has optional Attention"""
    def __init__(self, layer, in_channel, mlp, att=None, pna_agg=False):
        super(GACLayer_NoSampling, self).__init__()
        self.pna = pna_agg
        self.layer = layer
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.dropout = Config.dropout
        self.alpha = Config.alpha
        self.residual = None
        last_channel = in_channel  # the first last_channel == in_channel == 3 + 4 (pos + rgb,i)
        for out_channel in mlp:  # [32, 32, 64]
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel, track_running_stats=False))
            last_channel = out_channel
        self.group_all = Config.group_all
        if att == 'GAC':
            self.GAT = AttentionGAC(3+last_channel,last_channel,self.dropout,self.alpha)
        elif att == 'Mix':
            self.GAT = AttentionMix(10+last_channel, last_channel, self.dropout, self.alpha, self.pna)
        else:
            self.GAT = None

    def forward(self, new_xyz, new_points, grouped_xyz, fps_points):
        """
        Input:
            new_xyz: sampled (centre) points position (xyz), [B, npoint, C]
            new_points: sampled points feats (grouped), [B, npoint, nsample, C+D]
            grouped_xyz: sampled points position (xyz), [B, npoint, nsample, C]
            fps_points: sampled (centre) point feats [B, npoint, C+D,1]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample, npoint] x_j
        fps_points = fps_points.unsqueeze(3).permute(0, 2, 3, 1)  # [B, C+D, 1, npoint] x_i
        self.residual = copy.deepcopy(fps_points)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            fps_points = F.relu(bn(conv(fps_points)))
            new_points = F.relu(bn(conv(new_points)))
        '''new_points: [B, F, nsample,npoint]
           fps_points: [B, F, 1,npoint] '''
        if self.GAT is not None:
            new_points = self.GAT(center_xyz=new_xyz,
                                  center_feature=fps_points.squeeze().permute(0,2,1),
                                  grouped_xyz=grouped_xyz,
                                  grouped_feature=new_points.permute(0,3,2,1))
        else:
            new_points = torch.sum(new_points.permute(0, 3, 2, 1), dim=2)  # [B, npoint, D]; graph_pooling
        new_xyz = new_xyz.permute(0, 2, 1)
        new_points = new_points.permute(0, 2, 1)
        return new_xyz, new_points


class Res_Block(nn.Module):
    # adding residual to the GC layer
    def __init__(self, layer, mode, npoint, radius, nscale, in_channel,
                 out_channel, mlp, bn, act_fxn, att, pna_agg):
        super(Res_Block, self).__init__()
        self.mode = mode
        if self.mode == 'gc':
            self.body = GACLayer(npoint, radius, nscale, in_channel, mlp)
            self.mlp = SharedMLP(in_channel, out_channel, bn, act_fxn)  # for bringing fps_points up to hidden dim
        elif self.mode == 'gc_no_s':
            self.body = GACLayer_NoSampling(layer, in_channel, mlp, att, pna_agg)
            self.mlp = SharedMLP(in_channel, out_channel, bn, act_fxn)  # for bringing fps_points up to hidden dim
    def forward(self, xyz, points):
        if self.mode == 'gc':
            l_xyz, l_points = self.body(xyz, points)
        elif self.mode == 'gc_no_s':
            new_xyz, new_points, grouped_xyz, fps_points = max_dilation(xyz, points, layer=self.body.layer, sampler='fps')
            l_xyz, l_points = self.body(new_xyz, new_points, grouped_xyz, fps_points)
        
        residual = self.body.residual

        return l_xyz, l_points + self.mlp(residual).squeeze()  # chk to make sure dim match 


class Res_GACNet(nn.Module):
    # graph attention convolution with residual connections
    def __init__(self, mode):
        super(Res_GACNet, self).__init__()
        # conv, attention, (pna), residual
        # layer, mode, npoint, radius, nscale, in_channel, out_channel, mlp, bn, act_fxn, att, pna_agg
        self.res1 = Res_Block(0, mode, 1024, 0.3, Config.nscales[0], 3+3, 64, [32, 32, 64], True, False, 'Mix', True)
        self.res2 = Res_Block(1, mode, 256, 0.5, Config.nscales[1], 64+3, 128, [64, 64, 128], True, False, 'Mix', True)
        self.res3 = Res_Block(2, mode, 128, 0.8, Config.nscales[2], 128+3, 256, [128, 128, 256], True, False, 'Mix', True)
        self.res4 = Res_Block(3, mode, 64, 1.6, Config.nscales[3], 256+3, 512, [256, 256, 512], True, False, 'Mix', True)

        #pointNet feature propagation: in_channel, mlp
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

        # FC layers
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128, track_running_stats=False)
        self.drop1 = nn.Dropout(self.res1.body.dropout)
        self.conv2 = nn.Conv1d(128, Config.num_classes, 1)

    def forward(self, xyz, points):
        l1_xyz, l1_points = self.res1(xyz, points)
        l2_xyz, l2_points = self.res2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.res3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.res4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x


class GACNet_PNA(nn.Module):  
    # graph attention convolution with principal neigborhood aggregation
    def __init__(self, attention='Mix'):
        super(GACNet_PNA, self).__init__()
        # convs & GAT with PNA
        # GraphAttentionConvLayer_NoSampling: layer, in_channel, mlp, att=None, pna_agg=False
        self.sa1 = GACLayer_NoSampling(0, 3 + 3, [32, 32, 64], att=attention, pna_agg=True)
        self.sa2 = GACLayer_NoSampling(1, 64 + 3, [64, 64, 128], att=attention, pna_agg=True)
        self.sa3 = GACLayer_NoSampling(2, 128 + 3, [128, 128, 256], att=attention, pna_agg=True)
        self.sa4 = GACLayer_NoSampling(3, 256 + 3, [256, 256, 512], att=attention, pna_agg=True)

        #pointNet feature propagation: in_channel, mlp
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

        # FC layers
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128, track_running_stats=False)
        self.drop1 = nn.Dropout(self.sa1.dropout)
        self.conv2 = nn.Conv1d(128, Config.num_classes, 1)

    def forward(self, xyz, points):
        new_xyz, new_points, grouped_xyz, fps_points = max_dilation(xyz, points, layer=0, sampler='fps')
        l1_xyz, l1_points = self.sa1(new_xyz, new_points, grouped_xyz, fps_points)
        new_xyz, new_points, grouped_xyz, fps_points = max_dilation(l1_xyz, l1_points, layer=1, sampler='fps')
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        new_xyz, new_points, grouped_xyz, fps_points = max_dilation(l2_xyz, l2_points, layer=2, sampler='fps')
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        new_xyz, new_points, grouped_xyz, fps_points = max_dilation(l3_xyz, l3_points, layer=3, sampler='fps')
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x
