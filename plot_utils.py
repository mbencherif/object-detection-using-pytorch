import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import numpy as np
from PIL import Image
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import math


#subplot utils
def add_outline_path_effect(o, lw=4):
    o.set_path_effects([path_effects.Stroke(linewidth=lw, foreground='black'),
                       path_effects.Normal()])
    
def add_text_to_subplot(ax, pos, label, size='x-large', color='white'):
    text = ax.text(pos[0], pos[1], label, size=size, weight='bold', color=color, va='top')
    add_outline_path_effect(text, 2)
    
def hide_subplot_axes(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

def plot_image_tensor_in_subplot(ax, img_tensor):
    ima= img_tensor.cpu().numpy().transpose((1,2,0))
    ax.imshow(ima)

def show_img_in_subplot(ax, pil_image):
    im = np.array(pil_image)
    ax.imshow(im)
    
def draw_bbox(ax, bbox, color='white'): 
    #patches expects (x,y), w, h
    rect_patch = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],fill=False, lw=2, ec=color) 
    patch = ax.add_patch(rect_patch)
    add_outline_path_effect(patch, 4)
    
def plot_bbox_annotation(ax, bb, cat_label):
    draw_bbox(ax, bb)
    add_text_to_subplot(ax, (bb[0], bb[1]), cat_label)

def tensor_to_scalar(t):
    if t.dim()==0:
        return t.item()
    else:
        return t.numpy()    

#bbox utils    
def yxyx_to_xywh(ann):
    return [ann[1], ann[0], ann[3]-ann[1], ann[2]-ann[0]]

    
#plots used in largest item classifier 
def plot_trn_image_with_annotations(im_id, jpeg_dic, JPEG_DIR, annotations_dic, category_dic, figsize=(10, 10)):
    fig, ax = plt.subplots(1, figsize=figsize)
    show_img_in_subplot(ax, Image.open(JPEG_DIR/jpeg_dic[im_id]['file_name']))
    hide_subplot_axes(ax)
    
    annotations = [annotations_dic[im_id]] if(type(annotations_dic[im_id]) == tuple) else annotations_dic[im_id]
        
    for ann in annotations:
        plot_bbox_annotation(ax, ann[0], category_dic[ann[1]])
        
    plt.show()

def plot_horizontal_bar_chart(counts, labels, title='', x_tick_step=200):    
    sorted_items = sorted(zip(counts, labels), reverse=True)
    sorted_counts, sorted_labels = zip(*sorted_items)
    
    y_pos = np.arange(len(sorted_labels))
    
    fig, ax = plt.subplots(figsize=(15,8))
    ax.barh(y_pos, sorted_counts)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_labels)
    ax.set_xticks(range(0, int(sorted_counts[0]) + x_tick_step, x_tick_step))
    ax.invert_yaxis() 
    ax.set_facecolor('#f7f7f7')
    ax.set_title(title)
    
    for idx, val in enumerate(ax.patches):
        x_value = val.get_width() + 5
        y_value = 0.1 + val.get_y() + val.get_height()/2
        ax.text(x_value, y_value, int(sorted_counts[idx]))

    plt.show()        
    
def plot_model_predictions_on_sample_batch(batch, pred_labels, actual_labels, get_label_fn, n_items=12, plot_from=0, figsize=(16,12)):
    n_rows, n_cols = (1,n_items) if n_items<=4 else (math.ceil(n_items/4), 4)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    for i,ax in enumerate(axes.flat):
        plot_idx = plot_from + i
        plot_image_tensor_in_subplot(ax, batch[plot_idx])

        pred_label = get_label_fn(tensor_to_scalar(pred_labels[plot_idx])) 
        actual_label = get_label_fn(tensor_to_scalar(actual_labels[plot_idx]))  

        hide_subplot_axes(ax)
        add_text_to_subplot(ax, (0,0), 'Pred: '+pred_label)
        add_text_to_subplot(ax, (0,30), 'Actual: '+actual_label, color='yellow')

    plt.tight_layout()
    
       
# plots used in multi class classifier   
def add_bar_height_labels(ax, rects):
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    for rect in rects:
        height = int(rect.get_height())
        ax.text(rect.get_x() + rect.get_width()*offset['center'], 1.01*height,
                '{}'.format(height), ha=ha['center'], va='bottom')
          
def plot_class_wise_preds_gt_true_preds(predictions, actual_instances, correct_predictions, categories):    
    ind = np.arange(len(predictions))  # the x locations for the groups
    width = 0.3 # the width of the bars

    fig, ax = plt.subplots()
    fig.set_size_inches((25,10))

    rects1 = ax.bar(ind - width, predictions, width, 
                     label='Total class predictions')

    rects2 = ax.bar(ind, correct_predictions, width, 
                     label='Correct class predictions')
    
    rects3 = ax.bar(ind + width, actual_instances, width, 
                     label='Actual class instances')

    ax.set_xticks(ind)
    ax.set_xticklabels(categories)
    ax.legend()

    add_bar_height_labels(ax, rects1)
    add_bar_height_labels(ax, rects2)
    add_bar_height_labels(ax, rects3)
    plt.show()

def plot_class_precision_recall_curve(id, cats, ds_gt_label_logits, ds_pred_scores):    
    sk_y_true, sk_y_pred = ds_gt_label_logits[:,id].numpy(), ds_pred_scores[:,id].numpy()
    
    average_precision = average_precision_score(sk_y_true, sk_y_pred)
    precision, recall, thresholds = precision_recall_curve(sk_y_true, sk_y_pred)
    
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve for class "'+cats[id]+'" : AP={0:0.2f}'.format(average_precision))

def get_graph_data_for_multi_class_pr_curves(ds_gt_label_logits, ds_pred_scores, n_classes):    
    precision = dict()
    recall = dict()
    average_precision = dict()

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(ds_gt_label_logits[:, i].numpy(), ds_pred_scores[:, i].numpy())
        average_precision[i] = average_precision_score(ds_gt_label_logits[:, i].numpy(), ds_pred_scores[:, i].numpy())


    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(ds_gt_label_logits.numpy().ravel(), ds_pred_scores.numpy().ravel())
    average_precision["micro"] = average_precision_score(ds_gt_label_logits.numpy(), ds_pred_scores.numpy(), average="micro")

    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))
    
    return precision, recall, average_precision

def plot_average_precision_score_over_all_classes(precision, recall, average_precision):
    plt.figure()
    plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
        .format(average_precision["micro"]))
    plt.show()

def plot_precision_recall_curves_for_multi_class_labels(precision, recall, average_precision, cats):
    n_classes = len(cats)
    
    plt.figure(figsize=(20, 20))
    lines = []
    labels = []
    
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))

    
    for i in range(n_classes):
        color_RGB = np.random.rand(3)
        l, = plt.plot(recall[i], precision[i], color=color_RGB, lw=2, label=cats[i])
        lines.append(l)
        labels.append('"{0}" (area = {1:0.2f})'
                      ''.format(cats[i], average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall curves for all classes')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=15))

    plt.show()

def plot_precision_recall_vs_threshold_for_all_classes(n_classes, ds_gt_label_logits, ds_pred_scores, cats):
    n_rows, n_cols = (1, n_classes) if n_classes < 4 else (math.ceil(n_classes/4), 4)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 16))
    for i,ax in enumerate(axes.flat):
        precisions, recalls, thresholds = precision_recall_curve(ds_gt_label_logits[:, i].numpy(), ds_pred_scores[:,i].numpy())
        class_label = cats[i]

        ax.set_title('Class: "'+class_label+'"')
        ax.plot(thresholds, precisions[:-1], 'b--', label='precision')
        ax.plot(thresholds, recalls[:-1], 'g--', label = 'recall')
        ax.set_ylim([0,1])

        if i==0:
            ax.set_xlabel('Threshold')
            ax.legend(loc='upper left')

    plt.tight_layout()
    plt.show()    
    

#plots used in largest item bbox    
def plot_image_with_bbox(img_tensor, bbox_yxyx):
    bbox = yxyx_to_xywh(bbox_yxyx)
    
    fig, ax = plt.subplots(1)
    plot_image_tensor_in_subplot(ax, img_tensor)
    draw_bbox(ax, bbox)
    hide_subplot_axes(ax)
    plt.show()   
    
def plot_bbox_model_predictions_on_sample_batch(batch, pred_labels, actual_labels, n_items=12, plot_from=0, figsize=(16,12)):
    n_rows, n_cols = (1,n_items) if n_items<=4 else (math.ceil(n_items/4), 4) 
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    for i,ax in enumerate(axes.flat):
        plot_idx = plot_from + i
        
        pred_bbox = [int(x) for x in yxyx_to_xywh(pred_labels[plot_idx])]
        actual_bbox = yxyx_to_xywh(actual_labels[plot_idx])
        
        plot_image_tensor_in_subplot(ax, batch[plot_idx])
        draw_bbox(ax, pred_bbox)
        draw_bbox(ax, actual_bbox, color='yellow')
        
        add_text_to_subplot(ax, (pred_bbox[0], pred_bbox[1]), 'Pred:')
        add_text_to_subplot(ax, (actual_bbox[0], actual_bbox[1]), 'Actual:', color='yellow')
        
        hide_subplot_axes(ax)
  
    plt.tight_layout()    
    
    
#plots used in Concat model, largest item bbox plus classifier
def plot_concat_model_predictions_on_sample_batch(batch, pred_labels, actual_labels, get_label_fn, n_items=12, plot_from=0, figsize=(16,12)):
    pred_bboxes, pred_cat_ids = pred_labels
    actual_bboxes, actual_cat_ids = actual_labels
    
    n_rows, n_cols = (1,n_items) if n_items<=4 else (math.ceil(n_items/4), 4) 
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    for i,ax in enumerate(axes.flat):
        plot_idx = plot_from + i
        
        pred_bbox = [int(x) for x in yxyx_to_xywh(pred_bboxes[plot_idx])]
        actual_bbox = yxyx_to_xywh(actual_bboxes[plot_idx])
        
        plot_image_tensor_in_subplot(ax, batch[plot_idx])
        draw_bbox(ax, pred_bbox)
        draw_bbox(ax, actual_bbox, color='yellow')
        
        add_text_to_subplot(ax, (pred_bbox[0], pred_bbox[1]), 'Pred:'+get_label_fn(tensor_to_scalar(pred_cat_ids[plot_idx])))
        add_text_to_subplot(ax, (actual_bbox[0], actual_bbox[1]), 'Actual:'+get_label_fn(tensor_to_scalar(actual_cat_ids[plot_idx])) , color='yellow')
        hide_subplot_axes(ax)
  
    plt.tight_layout()

def plot_model_result_on_test_image(pred_bbox, pred_cat_id, get_label_fn, im_path):
    im = Image.open(im_path)
    w,h = im.size
    fig, ax = plt.subplots(1, 1)
    
    bbox = pred_bbox[0].clone()
    bbox = bbox/224
    bbox[0] = bbox[0]*h
    bbox[1] = bbox[1]*w
    bbox[2] = bbox[2]*h
    bbox[3] = bbox[3]*w
    
    pred_bbox = [int(x) for x in yxyx_to_xywh(bbox)]
    
    show_img_in_subplot(ax, im)
    draw_bbox(ax, pred_bbox)

    add_text_to_subplot(ax, (pred_bbox[0], pred_bbox[1]), 'Pred:'+get_label_fn(tensor_to_scalar(pred_cat_id[0])))
    hide_subplot_axes(ax)

    plt.tight_layout()
    plt.show()
