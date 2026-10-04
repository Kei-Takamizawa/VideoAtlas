from types import MethodType
import json
import numpy as np
import pytest
from videoatlas.model_adapters import SCRFDDetector, Detection, artifact_sha256
from videoatlas.recognition import RecognitionOptions


def test_multiscale_deduplicates_faces_restores_size_and_keeps_other_people():
    detector = object.__new__(SCRFDDetector)
    detector.size=(640,640); detector.strides=(8,16,32)
    detector.multiscale=True;detector.fixed_input=False
    seen=[]
    points=np.zeros((5,2),np.float32)
    def once(self,frame):
        seen.append(self.size)
        if self.size==(640,640): return [Detection((10,10,110,110),.8,points)]
        if self.size==(320,320): return [Detection((12,12,112,112),.9,points),Detection((150,10,240,110),.7,points)]
        return []
    detector._detect_once=MethodType(once,detector)
    detections=detector.detect(np.zeros((300,300,3),np.uint8))
    assert seen==[(640,640),(320,320),(960,960)]
    assert len(detections)==2 and detections[0].score==.9
    assert detector.size==(640,640)


def test_multiscale_restores_size_on_inference_failure():
    detector=object.__new__(SCRFDDetector)
    detector.size=(640,640);detector.strides=(8,16,32)
    detector.multiscale=True;detector.fixed_input=False
    def once(self,frame):
        if self.size==(320,320):raise RuntimeError("inference failed")
        return []
    detector._detect_once=MethodType(once,detector)
    with pytest.raises(RuntimeError,match="inference failed"):
        detector.detect(np.zeros((300,300,3),np.uint8))
    assert detector.size==(640,640)


def test_derived_detector_preserves_weights_and_nodes_and_checks_cache(tmp_path):
    onnx=pytest.importorskip("onnx")
    helper=onnx.helper
    source=tmp_path/'source.onnx'
    weight=onnx.numpy_helper.from_array(np.ones((1,),np.float32),name='unused_weight')
    graph=helper.make_graph([helper.make_node('Identity',['input'],['output'])],'metadata-test',
        [helper.make_tensor_value_info('input',onnx.TensorProto.FLOAT,[12800,1])],
        [helper.make_tensor_value_info('output',onnx.TensorProto.FLOAT,[12800,1])],[weight])
    original=helper.make_model(graph);onnx.save(original,str(source));before=source.read_bytes()
    options=RecognitionOptions(cache_dir=tmp_path/'cache')
    derived=SCRFDDetector._dynamic_output_model(source,options)
    changed=onnx.load(str(derived))
    assert source.read_bytes()==before
    assert [x.SerializeToString() for x in changed.graph.node]==[x.SerializeToString() for x in original.graph.node]
    assert [x.SerializeToString() for x in changed.graph.initializer]==[x.SerializeToString() for x in original.graph.initializer]
    assert changed.graph.output[0].type.tensor_type.shape.dim[0].dim_param
    metadata=json.loads(derived.with_suffix('.json').read_text())
    assert metadata['source_sha256']==artifact_sha256(source)
    assert metadata['derived_sha256']==artifact_sha256(derived)
    expected=derived.read_bytes();derived.write_bytes(b'corrupt')
    assert SCRFDDetector._dynamic_output_model(source,options).read_bytes()==expected
