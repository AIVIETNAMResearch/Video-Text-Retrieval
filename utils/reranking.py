import os
import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import albumentations
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import faiss
from collections import Counter
import h5py
from data_loader.dataset import LandmarkDataset, get_df, get_transforms


class Reranking1_Shoppe():
  def __init__(self,):
      pass

  def l2norm_numpy(self,x):
      return x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)

  def neighborhood_search(self,emb,thresh,k_neighbors):
      index = faiss.IndexFlatIP(emb.shape[1])
      faiss.normalize_L2(emb)
      index.add(emb)
      sim, I = index.search(emb, k_neighbors)
      pred_index=[]
      pred_sim=[]
      for i in range(emb.shape[0]):
          cut_index=0
          for j in sim[i]:
              if(j>thresh):
                  cut_index+=1
              else:
                  break                        
          pred_index.append(I[i][:(cut_index)])
          pred_sim.append(sim[i][:(cut_index)])
          
      return pred_index,pred_sim
    
  def blend_neighborhood(self,emb, match_index_lst, similarities_lst):
      new_emb = emb.copy()
      for i in range(emb.shape[0]):
          cur_emb = emb[match_index_lst[i]]
          weights = np.expand_dims(similarities_lst[i], 1)
          new_emb[i] = (cur_emb * weights).sum(axis=0)
      new_emb = self.l2norm_numpy(new_emb)
    
      return new_emb

  def iterative_neighborhood_blending(self,emb, threshes,k_neighbors):
      for thresh in threshes:
          match_index_lst, similarities_lst = self.neighborhood_search(emb, thresh,k_neighbors)
          emb = self.blend_neighborhood(emb, match_index_lst, similarities_lst)
      return emb, match_index_lst

class Reranking1_3_2019():
    def __init__(self, trainCSVPath):
        self.trainCSVPath = trainCSVPath
        pass

    def predict_landmark_id(self,ids_query, feats_query, ids_train, feats_train, landmark_dict, voting_k=3):
        print("\n")
        print('build index...')
        cpu_index = faiss.IndexFlatL2(feats_train.shape[1])
        cpu_index.add(feats_train)
        sims, topk_idx = cpu_index.search(x=feats_query, k=voting_k)
        print('query search done.')

        df = pd.DataFrame(ids_query, columns=['id'])
        images = []
        for idx in topk_idx:
          images.append(' '.join(ids_train[idx]))

        df['images'] = images
        rows = []
        for imidx, (_, r) in tqdm(enumerate(df.iterrows()), total=len(df)):
            image_ids = [name for name in r.images.split(' ')]
            counter = Counter()
            for i, image_id in enumerate(image_ids[:voting_k]):
                landmark_id = landmark_dict[image_id]

                counter[landmark_id] += sims[imidx, i]

            landmark_id, score = counter.most_common(1)[0]
            rows.append({
                'id': r['id'],
                'landmarks': f'{landmark_id} {score:.9f}',
            })

        pred = pd.DataFrame(rows).set_index('id')
        pred['landmark_id'], pred['score'] = list(
            zip(*pred['landmarks'].apply(lambda x: str(x).split(' '))))
        pred['score'] = pred['score'].astype(np.float32) / voting_k

        return pred


    def __call__(self,ids_index, feats_index,
                 ids_test, feats_test,
                 ids_train, feats_train,
                 subm, 
                 topk=100, voting_k=3, thresh=0.4):
        train19_csv = pd.read_csv(self.trainCSVPath)
        landmark_dict = train19_csv.set_index(
            'id').sort_index().to_dict()['landmark_id']

        pred_index = self.predict_landmark_id(
                ids_index, feats_index, ids_train, feats_train, landmark_dict, voting_k=voting_k)
        pred_test = self.predict_landmark_id(
            ids_test, feats_test, ids_train, feats_train, landmark_dict, voting_k=voting_k)

        assert np.all(subm['id'] == pred_test.index)
        subm['index_id_list'] = subm['images'].apply(lambda x: x.split(' ')[:topk])

        # to make higher score be inserted ealier position in insert-step
        pred_index = pred_index.sort_values('score', ascending=False)

        images = []
        for test_id, pred, ids in tqdm(zip(subm['id'], pred_test['landmark_id'], subm['index_id_list']),
                                            total=len(subm)):
            retrieved_pred = pred_index.loc[ids, ['landmark_id', 'score']]

            # Sort-step
            is_positive: pd.Series = (pred == retrieved_pred['landmark_id'])
            # use mergesort to keep relative order of original list.
            sorted_retrieved_ids: list = (~is_positive).sort_values(
                kind='mergesort').index.to_list()

            # Insert-step
            whole_positives = pred_index[pred_index['landmark_id'] == pred]
            whole_positives = whole_positives[whole_positives['score'] > thresh]
            # keep the order by referring to the original index
            # in order to insert the samples in descending order of score
            diff = sorted(set(whole_positives.index) - set(ids),
                          key=whole_positives.index.get_loc)
            pos_cnt = is_positive.sum()
            reranked_ids = np.insert(sorted_retrieved_ids, pos_cnt, diff)[:topk]

            images.append(' '.join(reranked_ids))

        subm['images'] = images

        return subm
class Reranking_3_2020:
    def __init__(self, batch_size, out_dim, CLS_TOP_K, TOP_K, data_dir, num_workers):
        self.batch_size = batch_size
        self.out_dim = out_dim
        self.CLS_TOP_K = CLS_TOP_K
        self.TOP_K = TOP_K
        self.data_dir = data_dir
        self.num_workers = num_workers
        
    def __call__(self,feats_train, test_loader, model, pred_mask,device="cuda"):
        if True:
          if isinstance(feats_train, np.ndarray):
            feats_train = torch.Tensor(feats_train).cuda()
          with torch.no_grad():
            PRODS = []
            PREDS = []
            PRODS_M = []
            PREDS_M = []   
            test_bar = tqdm(test_loader)   
            for img in test_bar:
              img = img.cuda()
              
              probs_m = torch.zeros([self.batch_size, self.out_dim],device=device)
              feat_b5,logits_m      = model(img)
              probs_m += logits_m
              
              feat = F.normalize(feat_b5)

              
              probs_m[:, pred_mask] += 1.0
              probs_m -= 1.0              

              (values, indices) = torch.topk(probs_m, self.CLS_TOP_K, dim=1)
              probs_m = values
              preds_m = indices              
              PRODS_M.append(probs_m.detach().cpu())
              PREDS_M.append(preds_m.detach().cpu())            
              
              distance = feat.mm(feats_train.T)
              (values, indices) = torch.topk(distance, self.TOP_K, dim=1)
              probs = values
              preds = indices    
              PRODS.append(probs.detach().cpu())
              PREDS.append(preds.detach().cpu())

            PRODS = torch.cat(PRODS).numpy()
            PREDS = torch.cat(PREDS).numpy()
            PRODS_M = torch.cat(PRODS_M).numpy()
            PREDS_M = torch.cat(PREDS_M).numpy()  
            
            torch.cuda.empty_cache()
            return PRODS, PREDS, PRODS_M,PREDS_M

def saveTrainFeatueToH5File(trainCSVPath, trainH5Path, model, transforms, data_dir, batch_size, num_workers):
  df_train = pd.read_csv(trainCSVPath)
  df_train['filepath'] = df_train['id'].apply(lambda x: os.path.join(data_dir,'train', '_'.join(x.split("_")[:-1]), f'{x}.jpg'))

  dataset_train = LandmarkDataset(df_train, 'test', 'test',transforms)
  train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers, shuffle = True)
  ids_train = df_train['id']
  max_len = len(max(ids_train, key = lambda x: len(x)))

  feats_train = []
  ids_train = df_train['id']
  for image in tqdm(train_loader):
    feat, _ = model(image.cuda())
    feat = feat.detach().cpu()
    feats_train.append(feat)

  feats_train = torch.cat(feats_train)
  feats_train = feats_train.cuda()
  feats_train = F.normalize(feats_train).cpu().detach().numpy().astype(np.float32)
  with h5py.File(trainH5Path, 'w') as f:
    f.create_dataset('ids', data=np.array(ids_train, dtype=f'S{max_len}'))
    f.create_dataset('feats', data=feats_train)

  torch.cuda.empty_cache()
  import gc
  gc.collect()
