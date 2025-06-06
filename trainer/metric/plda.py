# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: JFZhou 2019-11-18)
#           THU-CSTL  (Author: Lantian Li 2021-12-30)

import scipy
import numpy as np
import math
import os
from numba import jit
import sys
import logging

'''
 Modified from the code avaliable at https://github.com/vzxxbacq/PLDA/blob/master/plda.py
'''

# Logger
logger = logging.getLogger('libs')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

M_LOG_2PI = 1.8378770664093454835606594728112

class ClassInfo(object):

    def __init__(self, weight=0, num_example=0, mean=0):
        self.weight = weight
        self.num_example = num_example
        self.mean = mean

class PldaStats(object):

    def __init__(self, dim):
        self.dim_ = dim
        self.num_example = 0
        self.num_classes = 0
        self.class_weight = 0
        self.example_weight = 0
        self.sum = np.zeros([dim,1])
        self.offset_scatter= np.zeros([dim,dim])
        self.classinfo = list()

    def add_samples(self, weight, group):
        
        # Each row represent an utts of the same speaker.

        n = group.shape[0]
        mean = np.mean(group, axis=0)
        mean=mean.reshape((-1,1))

        self.offset_scatter += weight * np.matmul(group.T,group) 
        self.offset_scatter += -n * weight * np.matmul(mean,mean.T)
        self.classinfo.append(ClassInfo(weight, n, mean))

        self.num_example += n
        self.num_classes += 1
        self.class_weight += weight
        self.example_weight += weight * n
        self.sum += weight * mean

    def is_sorted(self):
        for i in range(self.num_classes-1):
            if self.classinfo[i+1].num_example < self.classinfo[i].num_example:
                return False
        return True

    def sort(self):

        for i in range(self.num_classes-1):
            for j in range(i+1,self.num_classes):
                if self.classinfo[i].num_example > self.classinfo[j].num_example: 
                    self.classinfo[i],self.classinfo[j] = self.classinfo[j],self.classinfo[i]

        return

class PLDA(object):

    def __init__(self, normalize_length=True, simple_length_norm=False):
        self.mean = 0
        self.transform = 0
        self.psi = 0
        self.dim = 0
        self.normalize_length = normalize_length
        self.simple_length_norm = simple_length_norm


    def transform_ivector(self,ivector, num_example):

        self.dim = ivector.shape[-1]
        transformed_ivec = self.offset
        transformed_ivec = 1.0 * np.matmul(self.transform ,ivector) + 1.0 * transformed_ivec

        if(self.simple_length_norm):
            normalization_factor = math.sqrt(self.dim) / np.linalg.norm(transformed_ivec)
        else:
            normalization_factor = self.get_normalization_factor(transformed_ivec,
                                                            num_example)
        if(self.normalize_length):
            transformed_ivec = normalization_factor*transformed_ivec

        return transformed_ivec


    def log_likelihood_ratio(self, transform_train_ivector, num_utts,
        transform_test_ivector):

        self.dim = transform_train_ivector.shape[0]
        mean = np.zeros([self.dim,1])
        variance = np.zeros([self.dim,1])
        for i in range(self.dim):
            mean[i] = num_utts * self.psi[i] / (num_utts * self.psi[i] + 1.0)*transform_train_ivector[i] #nΨ/(nΨ+I) u ̅^g
            variance[i] = 1.0 + self.psi[i] / (num_utts * self.psi[i] + 1.0)
        logdet = np.sum(np.log(variance)) #ln⁡|Ψ/(nΨ+I)+I|
        transform_test_ivector=transform_test_ivector.reshape(-1,1)
        sqdiff = transform_test_ivector - mean #u^p-nΨ/(nΨ+I) u ̅^g
        sqdiff=sqdiff.reshape(1,-1)
        sqdiff = np.power(sqdiff, 2.0)
        variance = np.reciprocal(variance)
        loglike_given_class = -0.5 * (logdet + M_LOG_2PI * self.dim + np.dot(sqdiff, variance))

        sqdiff = transform_test_ivector
        sqdiff = np.power(sqdiff, np.full(sqdiff.shape, 2.0))
        sqdiff=sqdiff.reshape(1,-1)
        variance = self.psi + 1.0
        logdet = np.sum(np.log(variance))
        variance = np.reciprocal(variance) #(Ψ+I)^(-1)
        variance=variance.reshape(-1,1)
        loglike_without_class = -0.5 * (logdet + M_LOG_2PI * self.dim + np.dot(sqdiff, variance))
        loglike_ratio = loglike_given_class - loglike_without_class

        return loglike_ratio


    def smooth_within_class_covariance(self, smoothing_factor):

        within_class_covar = np.ones(self.dim)
        smooth = np.full(self.dim,smoothing_factor*within_class_covar*self.psi.T)
        within_class_covar = np.add(within_class_covar,
                                    smooth)
        self.psi = np.divide(self.psi, within_class_covar)
        within_class_covar = np.power(within_class_covar,
                                    np.full(within_class_covar.shape, -0.5))
        self.transform = np.diag(within_class_covar) * self.transform
        self.compute_derived_vars()


    def compute_derived_vars(self):

        self.offset = np.zeros(self.dim)
        self.offset = -1.0 * np.matmul(self.transform,self.mean)
        return self.offset


    def get_normalization_factor(self, transform_ivector, num_example):

        transform_ivector_sq = np.power(transform_ivector, 2.0)
        inv_covar = self.psi + 1.0/num_example
        inv_covar = np.reciprocal(inv_covar)
        dot_prob = np.dot(inv_covar, transform_ivector_sq)

        return math.sqrt(self.dim/dot_prob)


    def compute_normalizing_transform(self,covar):

        c = np.linalg.cholesky(covar) 
        c = np.linalg.inv(c)

        return c

    def get_output(self):

        transform1 = self.compute_normalizing_transform(self.within_var) 
        '''
        // now transform is a matrix that if we project with it,
        // within_var_ becomes unit.
        // between_var_proj is between_var after projecting with transform1.
        '''
        between_var_proj =np.matmul(np.matmul(transform1 , self.between_var),transform1.T) 
        '''
        // Do symmetric eigenvalue decomposition between_var_proj = U diag(s) U^T,
        // where U is orthogonal.
        '''
        s, U = np.linalg.eig(between_var_proj)
        assert s.min()>0
        '''
        // The transform U^T will make between_var_proj diagonal with value s
        // (i.e. U^T U diag(s) U U^T = diag(s)).  The final transform that
        // makes within_var_ unit and between_var_ diagonal is U^T transform1,
        // i.e. first transform1 and then U^T.
        '''
        self.transform = np.matmul(U.T,transform1)
        self.psi = s
        self.compute_derived_vars()

    def plda_trans_write(self,plda):
      
        with open(plda,'w') as f:
            f.write('<Plda>  [ '+' '.join(list(map(str,list(self.mean.reshape(self.mean.shape[0])))))+' ]\n')
            f.write(' [')
            for i in range(len(self.transform)):    
                f.write('\n  '+' '.join(list(map(str,list(self.transform[i])))))
            f.write(' ]')
            f.write('\n [ '+' '.join(list(map(str,list(self.psi.reshape(self.psi.shape[0])))))+' ]\n')
            f.write('</Plda> ')


class PldaEstimation(object):

    def __init__(self, Pldastats):

        self.mean = 0

        self.stats = Pldastats
        is_sort = self.stats.is_sorted()
        if not is_sort:
            logger.info('The stats is not in order...')
            sys.exit()

        self.dim = Pldastats.dim_

        self.between_var =np.eye(self.dim)
        self.between_var_stats = np.zeros([self.dim,self.dim])
        self.between_var_count = 0
        self.within_var = np.eye(self.dim)
        self.within_var_stats = np.zeros([self.dim,self.dim])
        self.within_var_count = 0

    def estimate(self, num_em_iters = 10):
        for i in range(num_em_iters):
            logger.info("iteration times:{}".format(i))
            self.estimate_one_iter()
        self.mean = (1.0 / self.stats.class_weight) * self.stats.sum
    
    def estimate_one_iter(self):
        self.reset_per_iter_stats()
        self.get_stats_from_intraclass()
        self.get_stats_from_class_mean()
        self.estimate_from_stats()

    def reset_per_iter_stats(self):

        self.within_var_stats = np.zeros([self.dim,self.dim])
        self.within_var_count = 0
        self.between_var_stats = np.zeros([self.dim,self.dim])
        self.between_var_count = 0

    def get_stats_from_intraclass(self):

        self.within_var_stats += self.stats.offset_scatter
        self.within_var_count += (self.stats.example_weight - self.stats.class_weight)


    def get_stats_from_class_mean(self):

        within_var_inv = np.linalg.inv(self.within_var)
        between_var_inv =np.linalg.inv(self.between_var)

        for i in range(self.stats.num_classes):
            info = self.stats.classinfo[i]
            weight = info.weight
            if info.num_example:
                n = info.num_example
                mix_var = np.linalg.inv(between_var_inv +  n * within_var_inv) # [(Φ_b^(-1)+n[Φ_w]^(-1))]^(-1)
                m = info.mean - (self.stats.sum / self.stats.class_weight) # mk
                m=m.reshape((-1,1))
                temp = n * np.matmul(within_var_inv,m) # n[Φ_w]^(-1) m_k
                w = np.matmul(mix_var,temp) # w=[(Φ ̂)]^(-1) n[Φ_w]^(-1) m_k
                w=w.reshape(-1,1)
                m_w = m - w
                m_w=m_w.reshape(-1,1)
                self.between_var_stats += weight *mix_var #[(Φ_b^(-1)+n[Φ_w]^(-1))]^(-1)
                self.between_var_stats += weight *np.matmul(w,w.T) #[(Φ_b^(-1)+n[Φ_w]^(-1))]^(-1)+ww^T
                self.between_var_count += weight
                self.within_var_stats += weight * n * mix_var # n_k([(Φ_b^(-1)+n[Φ_w]^(-1))]^(-1)) 
                self.within_var_stats += weight *n *np.matmul(m_w,m_w.T)  # n_k([(Φ_b^(-1)+n[Φ_w]^(-1))]^(-1)) + (w_k-m_k) (w_k-m_k)^T
                self.within_var_count += weight

    def estimate_from_stats(self):

        self.within_var = (1.0 / self.within_var_count) * self.within_var_stats # 1/K ∑_k [n_k (Φ ̂_k+(w_k-m_k)[(w_k-m_k)]^T)]
        self.between_var = (1.0 / self.between_var_count) * self.between_var_stats # Φ_b=1/K ∑_k [(Φ ̂_k+w_k [w_k]^T)]


    def get_output(self):

        Plda_output = PLDA()
        # Plda_output.mean = (1.0 / self.stats.class_weight) * self.stats.sum
        Plda_output.mean =  self.mean
        transform1 = self.compute_normalizing_transform(self.within_var) # decomposition
        '''
        // now transform is a matrix that if we project with it,
        // within_var_ becomes unit.
        // between_var_proj is between_var after projecting with transform1.
        '''
        between_var_proj =np.matmul(np.matmul(transform1 , self.between_var),transform1.T)
        '''
        // Do symmetric eigenvalue decomposition between_var_proj = U diag(s) U^T,
        // where U is orthogonal.
        '''
        s, U = np.linalg.eig(between_var_proj)
        assert s.min()>0
        '''
        // The transform U^T will make between_var_proj diagonal with value s
        // (i.e. U^T U diag(s) U U^T = diag(s)).  The final transform that
        // makes within_var_ unit and between_var_ diagonal is U^T transform1,
        // i.e. first transform1 and then U^T.
        '''
        Plda_output.transform = np.matmul(U.T,transform1)
        Plda_output.psi = s
        Plda_output.compute_derived_vars()

        return Plda_output


    def compute_normalizing_transform(self,covar):
        c = np.linalg.cholesky(covar) 
        c = np.linalg.inv(c)
        return c

class PldaUnsupervisedAdaptor(object):
    def __init__(self,
                mean_diff_scale=1.0,
                within_covar_scale=0.3,
                between_covar_scale=0.7):
        self.tot_weight = 0
        self.mean_stats = 0
        self.variance_stats = 0
        self.mean_diff_scale = mean_diff_scale
        self.within_covar_scale = within_covar_scale
        self.between_covar_scale = between_covar_scale
    
    def add_stats(self, weight, ivector):
        ivector = np.reshape(ivector,(-1,1))
        if type(self.mean_stats)==int:
            self.mean_stats = np.zeros(ivector.shape)
            self.variance_stats = np.zeros((ivector.shape[0],ivector.shape[0]))
        self.tot_weight += weight
        self.mean_stats += weight * ivector
        self.variance_stats += weight * np.matmul(ivector,ivector.T)
        
    def update_plda(self, plda):
        dim = self.mean_stats.shape[0]
        '''
        // mean_diff of the adaptation data from the training data.  We optionally add
        // this to our total covariance matrix
        '''
        mean = (1.0 / self.tot_weight) * self.mean_stats
        '''
        D（x）= E[x^2]-[E(x)]^2
        '''
        variance = (1.0 / self.tot_weight) * self.variance_stats - np.matmul(mean,mean.T)
        '''
        // update the plda's mean data-member with our adaptation-data mean.
        '''
        mean_diff = mean - plda.mean
        variance += self.mean_diff_scale * np.matmul(mean_diff,mean_diff.T)

        plda.mean = mean
        transform_mod = plda.transform
        '''
        // transform_model_ is a row-scaled version_14 of plda->transform_ that
        // transforms into the space where the total covariance is 1.0.  Because
        // plda->transform_ transforms into a space where the within-class covar is
        // 1.0 and the the between-class covar is diag(plda->psi_), we need to scale
        // each dimension i by 1.0 / sqrt(1.0 + plda->psi_(i))
        '''
        for i in range(dim):
            transform_mod[i] *= 1.0 / math.sqrt(1.0 + plda.psi[i])
        '''
        // project the variance of the adaptation set into this space where
        // the total covariance is unit.
        '''
        variance_proj = np.matmul(np.matmul(transform_mod, variance),transform_mod.T)
        '''
        // Do eigenvalue decomposition of variance_proj; this will tell us the
        // directions in which the adaptation-data covariance is more than
        // the training-data covariance.
        '''
        s, P = np.linalg.eig(variance_proj)
        '''
        // W, B are the (within,between)-class covars in the space transformed by
        // transform_mod.
        '''
        W = np.zeros([dim, dim])
        B = np.zeros([dim, dim])
        for i in range(dim):
            W[i][i] = 1.0 / (1.0 + plda.psi[i])
            B[i][i] = plda.psi[i] / (1.0 + plda.psi[i])
        '''
        // OK, so variance_proj (projected by transform_mod) is P diag(s) P^T.
        // Suppose that after transform_mod we project by P^T.  Then the adaptation-data's
        // variance would be P^T P diag(s) P^T P = diag(s), and the PLDA model's
        // within class variance would be P^T W P and its between-class variance would be
        // P^T B P.  We'd still have that W+B = I in this space.
        // First let's compute these projected variances... we call the "proj2" because
        // it's after the data has been projected twice (actually, transformed, as there is no
        // dimension loss), by transform_mod and then P^T.
        '''
        Wproj2 = np.matmul(np.matmul(P.T, W),P)
        Bproj2 = np.matmul(np.matmul(P.T, B),P)
        Ptrans = P.T
        Wproj2mod = Wproj2
        Bproj2mod = Bproj2
        '''
        // For this eigenvalue, compute the within-class covar projected with this direction,
        // and the same for between.
        '''
        for i in range(dim):
            if s[i] > 1.0:
                excess_eig = s[i] - 1.0
                excess_within_covar = excess_eig * self.within_covar_scale
                excess_between_covar = excess_eig * self.between_covar_scale
                Wproj2mod[i][i] += excess_within_covar
                Bproj2mod[i][i] += excess_between_covar
        '''
        // combined transform "transform_mod" and then P^T that takes us to the space
        // where {W,B}proj2{,mod} are.
        '''
        combined_trans_inv = np.linalg.inv(np.matmul(Ptrans, transform_mod))
        '''
        // Wmod and Bmod are as Wproj2 and Bproj2 but taken back into the original
        // iVector space.
        '''
        Wmod = np.matmul(np.matmul(combined_trans_inv, Wproj2mod), combined_trans_inv.T)
        Bmod = np.matmul(np.matmul(combined_trans_inv, Bproj2mod), combined_trans_inv.T)

        '''
        // Do Cholesky Wmod = C C^T.  Now if we use C^{-1} as a transform, we have
        // C^{-1} W C^{-T} = I, so it makes the within-class covar unit.
        '''
        C_inv = np.linalg.inv(np.linalg.cholesky(Wmod))
        Bmod_proj = np.matmul(np.matmul(C_inv, Bmod), C_inv.T)
        '''
        // Do symmetric eigenvalue decomposition of Bmod_proj, so
        // Bmod_proj = Q diag(psi_new) Q^T
        '''
        psi_new, Q = np.linalg.eig(Bmod_proj)
        '''
        // This means that if we use Q^T as a transform, then Q^T Bmod_proj Q =
        // diag(psi_new), hence Q^T diagonalizes Bmod_proj (while leaving the
        // within-covar unit).
        // The final transform we want, that projects from our original
        // space to our newly normalized space, is:
        // first Cinv, then Q^T, i.e. the
        // matrix Q^T Cinv.
        '''
        final_transform = np.matmul(Q.T, C_inv)
        plda.transform = final_transform
        plda.psi = psi_new


class PldaAnalyzer(object):
    def __init__(self, n_components):
       self.plda_dim = n_components


    def fit(self, vector_data, spker_label, num_iter=2):
        self.global_mean = np.mean(vector_data, axis=0)
        vector_data = vector_data - self.global_mean
        spk2vec_dict = {}
        dim = np.shape(vector_data)[1]
        for i in range(len(vector_data)):
            spk = spker_label[i]
            if spk not in spk2vec_dict.keys():
                spk2vec_dict[spk] = np.reshape(vector_data[i], (-1, dim))
            else:
                spk2vec_dict[spk] = np.vstack((spk2vec_dict[spk], vector_data[i]))
        
        plda_stats = PldaStats(dim)
        for key in spk2vec_dict.keys():
            vectors = np.array(spk2vec_dict[key], dtype=float)
            weight = 1.0
            plda_stats.add_samples(weight,vectors)
        
        plda_stats.sort()
        plda_estimator = PldaEstimation(plda_stats)
        plda_estimator.estimate(num_em_iters=num_iter)
        self.plda = plda_estimator.get_output()
    
    def GetNormalizationFactor(self, transformed_vector, num_utts=1):
        assert(num_utts > 0)
        dim = len(transformed_vector)
        inv_covar = 1.0 / (1.0 / num_utts + self.plda.psi)
        dot_prod = np.dot(inv_covar, transformed_vector ** 2)
        return math.sqrt(dim / dot_prod)

    def TransformVector(self, vector, num_utts=1, simple_length_norm=True, normalize_length=True):
        dim = len(vector)
        normalization_factor = 0.0
        plda_b = self.plda.mean.ravel()
        plda_W = self.plda.transform.T
        plda_SB = self.plda.psi
        assert(len(plda_b)==len(plda_W))
        assert(len(plda_b)==len(plda_SB))
        transformed_vector = np.dot(vector - plda_b, plda_W )[:self.plda_dim]
        if simple_length_norm:
            normalization_factor = math.sqrt(dim) / np.linalg.norm(transformed_vector)
        else:
            normalization_factor = self.GetNormalizationFactor(transformed_vector)
        if normalize_length:
            transformed_vector = transformed_vector * normalization_factor
        return transformed_vector


    def transform(self, vectors, simple_length_norm=True, normalize_length=True):
        vectors = vectors - self.global_mean
        transformed_vectors = []
        for vector in vectors:
            transformed_vector = self.TransformVector(vector, simple_length_norm=simple_length_norm, normalize_length=normalize_length)
            transformed_vectors.append(transformed_vector)
        return np.array(transformed_vectors)

        
    def NLScore(self, enroll_vec, test_vec, enroll_num=1):
        '''
        normalized likelihood with uncertain means
        SB is the speaker between var
        SW is the speaker within var
        '''
        pi = np.array(np.pi)
        SB = self.plda.psi[:self.plda_dim]
        SW = 1
        uk = enroll_vec * (enroll_num * SB / (enroll_num * SB + SW))
        vk = SW + SB * SW / (enroll_num * SB + SW)
        pk = ((test_vec - uk)**2 / vk).sum() + np.log(2 * pi * vk).sum()
        px = (test_vec**2 / (SW + SB)).sum() + np.log(2 * pi * (SW + SB)).sum()

        score = 0.5 * (px - pk)
        return score

