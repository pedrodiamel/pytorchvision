
import numpy as np
import torch as th
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()        

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    

class AverageFilterMeter(AverageMeter):
    def __init__(self, depth=10):
        
        self.depth = depth
        self.buff = np.zeros( (self.depth,1) )
        self.favg = 0
        self.itr = 0
        self.n=0     

        super(AverageFilterMeter, self  ).__init__()

    def update(self, val, n=1):
        super(AverageFilterMeter, self).update( val, n)
        self.n+= int(self.n < self.depth)            
        self.buff[ self.itr ] = val
        self.favg = self.buff.sum()/self.n        
        self.itr = (self.itr+1)%len(self.buff)
        


class Logger(object):
    
    def __init__(self, title_name,  loss, metrics, plotter):
        """Initialization
        Args:
            @title_name: name id
            @loss: loss name list
            @metrics: metric name list
            @plotter: object plotter
        """
        self.plotter = plotter
        self.title_name = title_name

        dloss = dict(zip(loss, [ AverageFilterMeter() for _ in range(0, len(loss))]))
        dmetrics = dict(zip(metrics, [ AverageFilterMeter() for _ in range(0, len(metrics))]))

        self.info = {
            'loss': dloss ,
            'metrics': dmetrics ,
        }


    def _get( self ):        
        info = self.info
        for tag, value in info.items():
            for k,v in value.items():
                yield( tag, k, v )

    def update(self, loss, metrics, n):        
        for k,v in loss.items(): self.info['loss'][k].update( v, n )
        for k,v in metrics.items(): self.info['metrics'][k].update( v, n )

    
    def reset( self ):
        for t,k,v in self._get():
            v.reset()

    def logger(self, epoch, iterepoch, i, num, time, bplotter=True, bavg=False, bsummary=False  ): 

        strinfo  = '|{}: {:4d}|{:4d}|{:4d} '.format(self.title_name, epoch, i, num)
        strinfo += '|time: {:8.4f} '.format(time.val)  
        strsummary = '\nValidation:\n'       

        for t,k,v in self._get():
            strinfo += '|{}: {:8.4f} '.format( k, v.avg )
            strsummary += ' * {}: {:.3f} ({:.3f})\n'.format(k, v.val, v.avg )

            if not bplotter: continue 
            if bavg: self.plotter.plot(t, '{}_{} (avg)'.format(self.title_name,k), iterepoch, v.avg)
            else: self.plotter.plot(t, '{}_{} (avg)'.format(self.title_name,k), iterepoch, v.favg)
            self.plotter.plot(t, '{}_{} (val)'.format(self.title_name,k), iterepoch, v.val)

        print(strinfo, flush=True )
        if bsummary: print(strsummary, flush=True )





def image_summary(data):
    print(np.min(data), np.max(data), data.shape)

def summary(model, input_size):
    
        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split('.')[-1].split("'")[0]
                module_idx = len(summary)

                m_key = '%s-%i' % (class_name, module_idx+1)
                summary[m_key] = OrderedDict()
                summary[m_key]['input_shape'] = list(input[0].size())
                summary[m_key]['input_shape'][0] = -1
                summary[m_key]['output_shape'] = list(output.size())
                summary[m_key]['output_shape'][0] = -1

                params = 0
                if hasattr(module, 'weight'):
                    params += th.prod(th.LongTensor(list(module.weight.size())))
                    if module.weight.requires_grad:
                        summary[m_key]['trainable'] = True
                    else:
                        summary[m_key]['trainable'] = False
                if hasattr(module, 'bias'):
                    params +=  th.prod(th.LongTensor(list(module.bias.size())))
                summary[m_key]['nb_params'] = params
                
            if not isinstance(module, nn.Sequential) and \
               not isinstance(module, nn.ModuleList) and \
               not (module == model):
                hooks.append(module.register_forward_hook(hook))
                
        if th.cuda.is_available():
            dtype = th.cuda.FloatTensor
        else:
            dtype = th.FloatTensor
        
        # check if there are multiple inputs to the network
        if isinstance(input_size[0], (list, tuple)):
            x = [Variable(th.rand(1,*in_size)).type(dtype) for in_size in input_size]
        else:
            x = Variable(th.rand(1,*input_size)).type(dtype)
            
            
        # print(x.shape)
        # print(type(x[0]))
        # create properties
        summary = OrderedDict()
        hooks = []
        # register hook
        model.apply(register_hook)
        # make a forward pass
        model(x)
        # remove these hooks
        for h in hooks:
            h.remove()

        print('----------------------------------------------------------------')
        line_new = '{:>20}  {:>25} {:>15}'.format('Layer (type)', 'Output Shape', 'Param #')
        print(line_new)
        print('================================================================')
        total_params = 0
        trainable_params = 0
        for layer in summary:
            ## input_shape, output_shape, trainable, nb_params
            line_new = '{:>20}  {:>25} {:>15}'.format(layer, str(summary[layer]['output_shape']), summary[layer]['nb_params'])
            total_params += summary[layer]['nb_params']
            if 'trainable' in summary[layer]:
                if summary[layer]['trainable'] == True:
                    trainable_params += summary[layer]['nb_params']
            print(line_new)
        print('================================================================')
        print('Total params: ' + str(total_params))
        print('Trainable params: ' + str(trainable_params))
        print('Non-trainable params: ' + str(total_params - trainable_params))
        print('----------------------------------------------------------------')
        # return summary
