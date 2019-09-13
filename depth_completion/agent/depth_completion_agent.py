# builtin packages
import os
import numpy as np
from tqdm import tqdm

# torch
import torch
from torch import optim
from torch.utils.data import DataLoader

# from my module
from depth_completion.data import DepthDataset
from depth_completion.data import customed_collate_fn
import depth_completion.utils.loss_func as loss_func
from .base_agent import BaseAgent

class DepthCompletionAgent(BaseAgent):
    def __init__(self, config=None, file_manager=None):
        super(DepthCompletionAgent, self).__init__(config, file_manager)
        
        ###  Need to define your model yourself ###

        self.use_cuda = True if len(self._config['device_ids']) >= 1 else False
        self.device_ids = self._config['device_ids']
        if self.use_cuda is True:
            self.model = self.model.to(self.device_ids[0])
        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=self._config['lr'])  
        self._epoch = 0
        if self._config['load_model_path'] is not None:
            param_only = self._config['param_only']
            self.load_model(self._config['load_model_path'], param_only)
            if param_only == False:
                new_optimizer = optim.Adam(self.model.parameters(), 
                                            lr=self._config['lr'])  
                new_optimizer.load_state_dict(self.optimizer.state_dict())
                self.optimizer = new_optimizer
        print(self._epoch)
        print (self._config)

    def run(self):
        """Load data and start running model"""
        assert self._config['mode'] in ['train', 'test']
        dataset_name = self._config['dataset_name']
        # loss functions & weight
        # self.loss_funcs = list of (key (str), loss_func, weight (float))
        self.loss_funcs = []

        for loss_func_key, loss_func_name, weight in self._config['loss_func']:
            if not hasattr(loss_func, loss_func_name):
                raise AttributeError(f'Not supported loss function name: '\
                                     f'{loss_func_name}. Please add to '\
                                     f'utils/loss_func.py.')
            else:
                self.loss_funcs.append((loss_func_key,
                                        getattr(loss_func, 
                                                loss_func_name),
                                        weight))

        if self._config['mode'] == 'test':
            depth_test_dataset = DepthDataset(dataset_name, self._config['test_path'], train=False)
            self.test_loader = DataLoader(
                dataset=depth_test_dataset, 
                batch_size=self._config['batch_size'], 
                shuffle=False,
                num_workers=self._config['num_workers'],
                collate_fn=customed_collate_fn(dataset_name))
            # visualize output path
            if not os.path.isdir(self._config['output']):
                os.mkdir(self._config['output'])

            avg_loss, avg_detailed_loss = self.test()

        elif self._config['mode'] == 'train':
            # load datasets
            depth_dataset = DepthDataset(dataset_name, 
                                         self._config['train_path'], train=True)
            self.train_loader = DataLoader(
                dataset=depth_dataset, 
                batch_size=self._config['batch_size'], 
                shuffle=True, 
                num_workers=self._config['num_workers'],
                collate_fn=customed_collate_fn(dataset_name))

            if self._config['validation'] is True:
                depth_val_dataset = DepthDataset(
                    dataset_name, self._config['valid_path'], train=False)
                self.test_loader = DataLoader(
                    dataset=depth_val_dataset, 
                    batch_size=self._config['batch_size'], 
                    shuffle=False,
                    num_workers=self._config['num_workers'],
                    collate_fn=customed_collate_fn(dataset_name))

            # FileManager
            self._file_manager.set_base_path(self._config)
            self.train()

    def train(self):
        print ('Start Training ...')
        self.init_log()
        start_epoch = 0 if self._epoch == 0 else self._epoch + 1
        for self._epoch in range(start_epoch, self._config['epoches']):
            print(f'Start {self._epoch} epoch')
            # train
            self.train_loss, self.train_detailed_loss = self.train_one_epoch()
            # validation
            if self._config['validation'] is True:
                self.val_loss, self.val_detailed_loss = self.test(validate=True)
            # save log
            self.save_loss_to_log(print_msg=True)
            # save model
            self.save_model()

    def test(self, validate=False):
        tqdm_loader = tqdm(self.test_loader, total=len(self.test_loader))
        self.change_model_state('eval')

        # array to save loss for testing data
        total_valid_loss = []
        total_valid_detailed_loss = {}
        for (one_detailed_key, _, _) in self.loss_funcs:
            total_valid_detailed_loss[one_detailed_key] = []

        for step, batch in enumerate(tqdm_loader):
            with torch.no_grad():
                # go through network
                output = self.feed_into_net(batch, mode='eval')
                # calculate loss in single step
                loss, detailed_loss = self.calculate_loss(output, batch)
            
            total_valid_loss.append(loss.cpu().data.numpy())
            for one_detailed_key in detailed_loss.keys():
                total_valid_detailed_loss[one_detailed_key].append(
                    detailed_loss[one_detailed_key].cpu().data.numpy())
            
            # testing mode (visualize)
            if validate is False:
                scene_name = batch['scene_name']
                vis_items = {'original':batch['depth'], 
                             'output':output['ori_output_depth'], 
                             'render':batch['render_depth'],
                             'boundary':batch['depth_boundary']}
                self.save_image(vis_items, scene_name)
        tqdm_loader.close()

        # average loss
        avg_loss = np.mean(total_valid_loss)
        avg_detailed_loss = {key: np.mean(ele) for key, ele in total_valid_detailed_loss.items()}
        return avg_loss, avg_detailed_loss

    def train_one_epoch(self):
        tqdm_loader = tqdm(self.train_loader, total=len(self.train_loader))
        self.change_model_state('train')

        # array to save loss in one epoch
        total_train_loss = []
        total_detailed_loss = {}
        for (one_detailed_key, _, _) in self.loss_funcs:
            total_detailed_loss[one_detailed_key] = []

        for step, batch in enumerate(tqdm_loader):
            # go through network
            output = self.feed_into_net(batch, mode='train')
            # calculate loss in single step
            loss, detailed_loss = self.calculate_loss(output, batch)
            # backpropogation
            self.update(loss)
            total_train_loss.append(loss.cpu().data.numpy())
            for one_detailed_key in detailed_loss.keys():
                total_detailed_loss[one_detailed_key].append(
                    detailed_loss[one_detailed_key].cpu().data.numpy())
            if self._epoch == 0:
                _avg_loss = np.mean(total_train_loss)
                print (f'step : {step}, loss : {_avg_loss}')
                for one_detailed_key in detailed_loss.keys():
                    print (f'{one_detailed_key} : {detailed_loss[one_detailed_key].cpu().data.numpy()}',
                            end=" ")
                print("")
        tqdm_loader.close()
        # average loss
        avg_loss = np.mean(total_train_loss)
        avg_detailed_loss = {key: np.mean(ele) for key, ele in total_detailed_loss.items()}
        return avg_loss, avg_detailed_loss

    def feed_into_net(self, batch, mode):
        assert mode in {'train', 'eval'}
        # load into GPU
        if self.use_cuda: 
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device_ids[0])
        # process batch data
        color = batch['color']
        depth = batch['depth']
        normal = batch['normal']
        mask = batch['mask']
        boundary = batch['boundary']
        render_depth = batch['render_depth']

        # extract feature and feed into model
        feature = torch.cat([color, depth, normal, boundary, mask], dim=1)
        if self.use_cuda and mode == 'train':
            output_depth = torch.nn.parallel.data_parallel(
                self.model, feature, device_ids=self.device_ids)
        else:
            # Avoid SN in eval mode crash
            output_depth = self.model(feature)
        
        # process output
        output_mask = None
        render_depth_mask = torch.ones_like(render_depth)
        render_depth_mask[render_depth == 0] = 0
        ori_output_depth = output_depth
        output_depth = ori_output_depth * render_depth_mask
        output = {'output_depth': output_depth, 
                  'ori_output_depth': ori_output_depth}
        return output

    def calculate_loss(self, output, batch):
        """Calculate loss based on output and ground truth
        return:
            loss (torch.Tensor): the total loss calculated 
            detailed_loss ({'loss_name': loss}): detailed loss with loss name
        """

        detailed_loss = {}
        for loss_func_key, this_loss_func, weight in self.loss_funcs:
            this_loss = this_loss_func(output, batch) * weight
            detailed_loss[loss_func_key] = this_loss
        loss = sum(detailed_loss.values())
        return loss, detailed_loss

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def init_log(self):
        def _append_log_order_msg(log_msg, valid):
            for (loss_func_key, _, _) in self.loss_funcs:
                if valid:
                    loss_func_key = 'valid_' + loss_func_key
                self._log_order.append(loss_func_key)
                log_msg += ',%s' % loss_func_key
            return log_msg
        self._log_order = ['epoch', 'loss']
        log_msg = 'epoch,loss'
        if len(self.loss_funcs) > 1:
            log_msg = _append_log_order_msg(log_msg, False)
        self._log_order.append('valid_loss')
        log_msg += ',valid_loss'
        if len(self.loss_funcs) > 1 and self._config['validation']:
            log_msg = _append_log_order_msg(log_msg, True)
        self.save_log(log_msg)

    def save_loss_to_log(self, print_msg=True):
        log_msg = ''
        for order_name in self._log_order:
            if order_name == 'epoch':
                log_msg += f'{self._epoch}'
                continue
            if 'valid_' in order_name and 'valid_' == order_name[:6]:
                if order_name == 'valid_loss':
                    log_msg += ',%.3f' % self.val_loss
                else:
                    key_name = order_name[6:]
                    log_msg += ',%.3f' % self.val_detailed_loss[key_name]
            else:
                if order_name == 'loss':
                    log_msg += ',%.3f' % self.train_loss
                else:
                    log_msg += ',%.3f' % self.train_detailed_loss[order_name]
        if print_msg:
            print(','.join(self._log_order))
            print(log_msg)
        self.save_log(log_msg)

    def change_model_state(self, state):
        if state == 'train':
            self.model.train()
        elif state == 'eval':
            self.model.eval()
