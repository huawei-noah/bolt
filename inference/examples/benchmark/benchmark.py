import bolt
import argparse
 
parser = argparse.ArgumentParser(description='Bolt benchmark. If you want to profiling network and get execution time of each layer, please rebuild Bolt with --profile option.')
parser.add_argument('-m', type=str, required=True, dest="boltModelPath", help="Bolt model file path on disk.")
parser.add_argument('-i', type=str, dest="inputDataPath", help="Input data file path on disk.")
parser.add_argument('-a', type=str, dest="affinityPolicyName", default="CPU_HIGH_PERFORMANCE", help="Affinity policy", choices=['CPU_HIGH_PERFORMANCE','CPU_LOW_POWER','GPU','CPU'])
parser.add_argument('-p', type=str, dest="algorithmMapPath", help="Algorithm configration path.")
parser.add_argument('-l', type=int, default=10, dest="loopTime", help="Loop running times.")
parser.add_argument('-w', type=int, default=10, dest="warmTime", help="Warm up times.")
parser.add_argument('-t', type=int, default=10, dest="threadsNum", help="Parallel threads num.")

args = parser.parse_args()

net = bolt.Bolt()

net.set_num_threads(args.threadsNum)

net.load(args.boltModelPath, args.affinityPolicyName)

input_info = net.get_input_info()

data = {}
for k, v in input_info.items():
    num = 1
    for i in v:
        num = num * i
    data[k] = [1.0] * num

output = net.infer(data)
print(net.get_output_info())
