from __future__ import absolute_import
from . import caffe_pb2 as pb
import google.protobuf.text_format as text_format
import numpy as np

from .layer_parameter import LayerParameter


class _Net(object):
    def __init__(self):
        self.net = pb.NetParameter()
        self.name_dict = {}
        self.add_layer_set = True

    def set_add_layer(self, add_layer_set):
        self.add_layer_set = add_layer_set

    def layer_index(self, layer_name):
        for i, layer in enumerate(self.net.layer):
            if layer.name == layer_name:
                return i
        return -1

    def add_output(self, outputs):
        for item in outputs:
            self.net.output.append(item)

    def add_layer(self, layer_params, before='', after=''):
        if (not self.add_layer_set):
            return

        if (layer_params.layerName in self.name_dict):
            print("[ERROR] layer %s duplicate" % (layer_params.layerName))
            exit(-1)
        else:
            self.name_dict[layer_params.layerName] = 1
        index = -1
        if after != '':
            index = self.layer_index(after) + 1
        if before != '':
            index = self.layer_index(before)
        new_layer = pb.LayerParameter()
        new_layer.CopyFrom(layer_params.layerParameter)
        if index != -1:
            self.net.layer.add()
            for i in xrange(len(self.net.layer) - 1, index, -1):
                self.net.layer[i].CopyFrom(self.net.layer[i - 1])
            self.net.layer[index].CopyFrom(new_layer)
        else:
            self.net.layer.extend([new_layer])


    def remove_layer_by_name(self, layer_name):
        for i,layer in enumerate(self.net.layer):
            if layer.name == layer_name:
                del self.net.layer[i]
                return


    def get_layer_by_name(self, layer_name):
        for layer in self.net.layer:
            if layer.name == layer_name:
                return layer


    def save_prototxt(self, path):
        prototxt = pb.NetParameter()
        prototxt.CopyFrom(self.net)
        for layer in prototxt.layer:
            del layer.blobs[:]
        with open(path,'w') as f:
            f.write(text_format.MessageToString(prototxt))


    def layer(self, layer_name):
        return self.get_layer_by_name(layer_name)


    def layers(self):
        return list(self.net.layer)



class Prototxt(_Net):
    def __init__(self, file_name=''):
        super(Prototxt, self).__init__()
        self.file_name = file_name
        if file_name != '':
            f = open(file_name,'r')
            text_format.Parse(f.read(), self.net)
            pass

    def init_caffemodel(self,caffe_cmd_path='caffe'):
        s = pb.SolverParameter()
        s.train_net = self.file_name
        s.max_iter = 0
        s.base_lr = 1
        s.solver_mode = pb.SolverParameter.CPU
        s.snapshot_prefix = './nn'
        with open('/tmp/nn_tools_solver.prototxt','w') as f:
            f.write(str(s))
        os.system('%s train --solver /tmp/nn_tools_solver.prototxt'%caffe_cmd_path)


class CaffeModel(_Net):
    def __init__(self, file_name=''):
        super(CaffeModel, self).__init__()
        if file_name != '':
            f = open(file_name,'rb')
            self.net.ParseFromString(f.read())
            f.close()


    def save(self, path):
        with open(path,'wb') as f:
            f.write(self.net.SerializeToString())


    def add_layer_with_data(self,layer_params,datas, before='', after=''):
        self.add_layer(layer_params,before,after)
        new_layer = self.layer(layer_params.name)

        #process blobs
        del new_layer.blobs[:]
        for data in datas:
            new_blob = new_layer.blobs.add()
            for dim in data.shape:
                new_blob.shape.dim.append(dim)
            new_blob.data.extend(data.flatten().astype(float))


    def get_layer_data(self, layer_name):
        layer = self.layer(layer_name)
        datas = []
        for blob in layer.blobs:
            shape = list(blob.shape.dim)
            data = np.array(blob.data).reshape(shape)
            datas.append(data)
        return datas


    def set_layer_data(self, layer_name, datas):
        layer = self.layer(layer_name)
        for blob,data in zip(layer.blobs,datas):
            blob.data[:] = data.flatten()
            pass
