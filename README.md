'### Training

1) Change file path to word embedding in data_utils.py, method build_embedding_matrix, fname =  
2) python train.py --dataset='psytar'

### Add new corpus
1) In data_utils.py, class ABSADatesetReader, method __init__ add paths to train and test files in self.fname dict
2) Count max context len, add to context_dict in train.py file