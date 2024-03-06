import onnxruntime as rt

def onnx_infer_det(img, onnx_path):
    sess = rt.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    output_name0 = sess.get_outputs()[0].name
    # output_name1 = sess.get_outputs()[1].name
    pred = sess.run([output_name0], {input_name: img})[0]  # (bs, 84=80cls+4reg, 8400=3种尺度的特征图叠加)， 这里的预测框的回归参数是xywh，而不是中心点到框边界的距离
    # cls_pred = sess.run([output_name1], {input_name: img})[0]
    # pred = np.concatenate((bbox_pred, cls_pred), 2)
    return pred

def onnx_infer_pose(img, onnx_path):
    sess = rt.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    output_name0 = sess.get_outputs()[0].name
    output_name1 = sess.get_outputs()[1].name
    simcc_x = sess.run([output_name0], {input_name: img})[0]
    simcc_y = sess.run([output_name1], {input_name: img})[0]
    return simcc_x, simcc_y