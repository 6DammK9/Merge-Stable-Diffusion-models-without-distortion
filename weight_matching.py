import torch
from scipy.optimize import linear_sum_assignment
import time
import random
from merge_PermSpec_ResNet import mlp_permutation_spec
from PermSpec_Base import PermutationSpec
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

def get_permuted_param(ps: PermutationSpec, perm, k: str, params, except_axis=None):
  """Get parameter `k` from `params`, with the permutations applied."""
  w = params[k]
  # Printing on screen will make the process very slow. Don't leave it on in final version
  #print(k)
  
  # I will remove the try block also. Rewrite it when needed.
  for axis, p in enumerate(ps.axes_to_perm[k]):
    # Skip the axis we're trying to permute.
    if axis == except_axis:
      continue

    # None indicates that there is no permutation relevant to that axis.
    if p is not None:
      w = torch.index_select(w, axis, perm[p].int())

  return w

def apply_permutation(ps: PermutationSpec, perm, params):
  """Apply a `perm` to `params`."""
  return {k: get_permuted_param(ps, perm, k, params) for k in params.keys() if "model_" not in k}

def weight_matching(ps: PermutationSpec, params_a, params_b, special_layers=None, device="cpu", max_iter=10, init_perm=None, usefp16=False, workers=1):
  """Find a permutation of `params_b` to make them match `params_a`."""
  # tqdm layer will start from 1.
  
  perm_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items() if axes[0][0] in params_b}
  #print(perm_sizes)
  perm = dict()
  perm = {p: torch.arange(n) for p, n in perm_sizes.items()} if init_perm is None else init_perm
  special_layers = special_layers if special_layers and len(special_layers) > 0 else sorted(list(perm.keys()))
  #print(special_layers)
  sum_loss = 0.0 
  sum_number = 0
  EPS = 1e-12

  def make_increment_A_fp16(wk, axis, n):
    w_a = params_a[wk]
    w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)
    w_a = torch.moveaxis(w_a, axis, 0).reshape((n, -1)).to(device)
    w_b = torch.moveaxis(w_b, axis, 0).reshape((n, -1)).T.to(device)
    #A += torch.matmul(w_a.half(), w_b.half())
    return torch.matmul(w_a.half(), w_b.half())

  def make_increment_A_fp32(wk, axis, n):
    w_a = params_a[wk]
    w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)
    w_a = torch.moveaxis(w_a, axis, 0).reshape((n, -1)).to(device)
    w_b = torch.moveaxis(w_b, axis, 0).reshape((n, -1)).T.to(device)
    #A += torch.matmul(w_a.float(), w_b.float()).cpu()
    return torch.matmul(w_a.float(), w_b.float()).cpu()

  def update_perm_fp16(p_ix):
    progress = True
    p = p_ix
    loss = 0.0
    number = 0
    if p in special_layers:
      n = perm_sizes[p]
      iter_a = [make_increment_A_fp16(wk, axis, n) for wk, axis in ps.perm_to_axes[p]]
      A = torch.stack(iter_a, dim=0).sum(dim=0).cpu()

      ri, ci = linear_sum_assignment(A.detach().numpy(), maximize=True)

      assert (torch.tensor(ri) == torch.arange(len(ri))).all()
      
      oldL = torch.vdot(torch.flatten(A).float(), torch.flatten(torch.eye(n)[perm[p].long()]).float()).half()
      newL = torch.vdot(torch.flatten(A).float(), torch.flatten(torch.eye(n)[ci, :]).float()).half()
      
      if newL - oldL != 0:
        #sum += abs((newL-oldL).item())
        #number += 1
        loss = abs((newL-oldL).item())
        number = 1
        #print(f"{p}: {newL - oldL}")

      progress = progress or newL > oldL + EPS

      perm[p] = torch.Tensor(ci)
    return {
      "progress": progress,
      "loss": loss,
      "number": number
    }

  def update_perm_fp32(p_ix):
    progress = False
    p = p_ix
    loss = 0.0
    number = 0
    if p in special_layers:
      n = perm_sizes[p]
      iter_a = [make_increment_A_fp32(wk, axis, n) for wk, axis in ps.perm_to_axes[p]]
      A = torch.stack(iter_a, dim=0).sum(dim=0).cpu()
      ri, ci = linear_sum_assignment(A.detach().numpy(), maximize=True)

      assert (torch.tensor(ri) == torch.arange(len(ri))).all()
    
      oldL = torch.vdot(torch.flatten(A), torch.flatten(torch.eye(n)[perm[p].long()]).float())
      newL = torch.vdot(torch.flatten(A), torch.flatten(torch.eye(n)[ci, :]).float())

      if newL - oldL != 0:
        #sum += abs((newL-oldL).item())
        #number += 1
        loss = abs((newL-oldL).item())
        number = 1
        #print(f"{p}: {newL - oldL}")

      progress = progress or newL > oldL + EPS

      perm[p] = torch.Tensor(ci)
    return {
      "progress": progress,
      "loss": loss,
      "number": number
    }

  pbar = tqdm(range(max_iter), desc="weight_matching", position=1)
  for _ in pbar:
    #progress = False
    random.shuffle(special_layers)
    perm_arr = []
    if usefp16:
      #for p_ix in tqdm(special_layers, desc="weight_matching for special_layers", position=2):
      perm_arr = thread_map(update_perm_fp16, special_layers, desc="weight_matching for special_layers", position=2, max_workers=workers)
    else:
      perm_arr = thread_map(update_perm_fp32, special_layers, desc="weight_matching for special_layers", position=2, max_workers=workers)
        
    progress_arr = [d["progress"] for d in perm_arr]
    sum_loss += sum([d["loss"] for d in perm_arr])
    sum_number += sum([d["number"] for d in perm_arr])

    pbar.set_postfix({'sum_loss': sum_loss, 'number': sum_number})

    #if not progress:
    if not (True in progress_arr):
      break
  
  if sum_number > 0:
    average = sum_loss / sum_number
  else:
    average = 0
  #pbar.set_postfix({'average': average})
  return (perm, average)

def test_weight_matching():
  """If we just have a single hidden layer then it should converge after just one step."""
  ps = mlp_permutation_spec(num_hidden_layers=3)
  #print(ps.axes_to_perm)
  rng = torch.Generator()
  rng.manual_seed(13)
  num_hidden = 10
  shapes = {
      "layer0.weight": (2, num_hidden),
      "layer0.bias": (num_hidden, ),
      "layer1.weight": (num_hidden, 3),
      "layer1.bias": (3, )
  }

  rngmix = lambda rng, x: random.fold_in(rng, hash(x))

  params_a = {k: random.normal(rngmix(rng, f"a-{k}"), shape) for k, shape in shapes.items()}
  params_b = {k: random.normal(rngmix(rng, f"b-{k}"), shape) for k, shape in shapes.items()}
  perm = weight_matching(rng, ps, params_a, params_b)
  print(perm)

if __name__ == "__main__":
  test_weight_matching()
