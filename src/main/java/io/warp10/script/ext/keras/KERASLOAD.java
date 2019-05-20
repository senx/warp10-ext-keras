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

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;

import io.warp10.WarpConfig;
import io.warp10.script.NamedWarpScriptFunction;
import io.warp10.script.WarpScriptException;
import io.warp10.script.WarpScriptStack;
import io.warp10.script.WarpScriptStackFunction;

/**
 * Load a Keras model.
 * For the list of supported Keras features,
 * @see https://deeplearning4j.org/docs/latest/keras-import-supported-features
 */
public class KERASLOAD extends NamedWarpScriptFunction implements WarpScriptStackFunction {
  public KERASLOAD(String name) {
    super(name);
  }

  @Override
  public Object apply(WarpScriptStack stack) throws WarpScriptException {
    
    Object top = stack.pop();
    
    if (!(top instanceof String) && !(top instanceof byte[])) {
      throw new WarpScriptException(getName() + " expects a model (BYTES) or model file name (STRING).");
    }
    
    InputStream in = null;
        
    try {
      if (top instanceof byte[]) {
        in = new ByteArrayInputStream((byte[]) top);
      } else {
        File root = KerasWarpScriptExtension.getRoot();
        
        if (null == root && !KerasWarpScriptExtension.useClasspath()) {
          throw new WarpScriptException(getName() + " cannot load model from file, model root directory was not set.");
        }
        
        File modelfile = null;
        
        if (null != root) {
          modelfile = new File(root, String.valueOf(top));
        
          if (!modelfile.getAbsolutePath().startsWith(root.getAbsolutePath())) {
            throw new WarpScriptException(getName() + " invalid path for model.");
          }        
        }
              
        if (null == modelfile || !modelfile.exists()) {
          if (!KerasWarpScriptExtension.useClasspath()) {
            throw new WarpScriptException(getName() + " model could not be loaded.");
          }
          
          in = WarpConfig.class.getClassLoader().getResourceAsStream(String.valueOf(top));
          if (null == in) {
            throw new WarpScriptException(getName() + " model could not be found in class path ");
          }
        } else {
          in = new FileInputStream(modelfile);
        }
      }

      ComputationGraph cg = KerasModelImport.importKerasModelAndWeights(in);
      
      stack.push(cg);
    } catch (UnsupportedKerasConfigurationException ukce) {
      throw new WarpScriptException(getName() + " could not load model.", ukce);    
    } catch (InvalidKerasConfigurationException ikce) {
      throw new WarpScriptException(getName() + " could not load model.", ikce);
    } catch (IOException ioe) {
      throw new WarpScriptException(getName() + " could not load model.", ioe);      
    } finally {
      if (null != in) {
        try { in.close(); } catch (IOException ioe) {}
      }
    }

    return stack;
  }
}
