//
//   Copyright 2019  SenX S.A.S.
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//

package io.warp10.script.ext.keras;

import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import io.warp10.script.NamedWarpScriptFunction;
import io.warp10.script.WarpScriptException;
import io.warp10.script.WarpScriptStack;
import io.warp10.script.WarpScriptStackFunction;

public class KERASEVAL extends NamedWarpScriptFunction implements WarpScriptStackFunction {
  public KERASEVAL(String name) {
    super(name);
  }

  @Override
  public Object apply(WarpScriptStack stack) throws WarpScriptException {
    
    Object top = stack.pop();
    
    if (!(top instanceof List)) {
      throw new WarpScriptException(getName() + " expects a list of arrays as parameter.");
    }
    
    List<Object> arrays = (List<Object>) top;
    
    top = stack.pop();
    
    if (!(top instanceof ComputationGraph)) {
      throw new WarpScriptException(getName() + " operates on a Keras model.");
    }
    
    ComputationGraph cg = (ComputationGraph) top;
    
    if (arrays.size() != cg.getNumInputArrays()) {
      throw new WarpScriptException(getName() + " Keras model expects " + cg.getNumInputArrays() + " input arrays, found " + arrays.size());
    }
    
    INDArray[] inputs = cg.getInputs();
    
    if (null == inputs) {
      inputs = new INDArray[cg.getNumInputArrays()];
    }
    
    for (int i = 0; i < inputs.length; i++) {
      Object array = arrays.get(i);
      
      if (!(array instanceof List)) {
        throw new WarpScriptException(getName() + " expects inputs to be arrays.");
      }
      
      //
      // Determine dimensions
      //
      
      List<Integer> shape = new ArrayList<Integer>();
      
      List<Object> l = (List<Object>) array;
      
      shape.add(l.size());
      
      while(l.get(0) instanceof List) {
        l = (List<Object>) l.get(0);
        shape.add(l.size());
      }
      
      l = (List<Object>) array;
      
      int[] shapea = new int[shape.size()];
      
      long ncells = 1;

      for (int k = 0; k < shapea.length; k++) {
        shapea[k] = shape.get(k).intValue();
        ncells = ncells * shapea[k];
      }
      
      inputs[i] = Nd4j.create(ncells);
      int[] indices = new int[shapea.length];
      
      long count = 0;
      
      Object value = null;
      
      while(count < ncells) {
        
        Object o = l;
        
        for (int j = 0; j < indices.length; j++) {
          o = ((List<Object>) o).get(indices[j]);
          if (j == indices.length - 1) {
            value = o;
          }
        }
                
        if (!(value instanceof Number)) {
          throw new WarpScriptException(getName() + " encountered a non numeric input.");
        }

        inputs[i].putScalar(count, ((Number) value).doubleValue());  
        count++;
        
        for (int j = indices.length - 1; j >= 0; j--) {
          indices[j]++;
          if (indices[j] >= shapea[j]) {
            indices[j] = 0;
          } else {
            break;
          }
        }
      }      
      
      inputs[i] = inputs[i].reshape(shapea);
    }
    
    INDArray[] outputs = cg.output(inputs);
    
    List<Object> results = new ArrayList<Object>(outputs.length);

    for (int i = 0; i < outputs.length; i++) {
      long[] shape = outputs[i].shape();
      long ncells = 1;
      for (long l: shape) {
        ncells = ncells * l;
      }

      outputs[i] = outputs[i].reshape(ncells);
      
      List<Object> l = new ArrayList<Object>((int) ncells);
      
      // Copy the elements
      
      for (int k = 0; k < ncells; k++) {
        l.add(outputs[i].getDouble(k));
      }
      
      // Now perform a reshape of the List

      for (int k = shape.length - 1; k >= 0; k--) {
        int size = (int) shape[k];
        List<Object> ll = l;
        l = new ArrayList<Object>(l.size() / size);
        int idx = 0;
        while (idx < ll.size()) {
          l.add(ll.subList(idx, idx + size));
          idx += size;
        }
      }

      results.add(l);
    }
    
    stack.push(results);
        
    return stack;
  }
}
