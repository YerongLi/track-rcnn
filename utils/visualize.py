import os 
import numpy as np
import time
import imp
import cv2

import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Visualization functions
# Color map
cmap = matplotlib.cm.get_cmap('bwr')
min_score = 0.
max_score = 1.
norm = matplotlib.colors.Normalize(min_score, max_score)
    
def plot_box(ax, box, score, visible=True, show_score=True):
    """
    Plot an roi on ax.
    """
    x1, y1, x2, y2 = box.astype('float32')
    
    if score is None:
        c = 'r'
    else:
        c = cmap(norm(score))
        
    if visible:
        linestyle = 'solid'
    else:
        linestyle = 'dashed'
        
    ax.add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, linewidth=1,
                linestyle=linestyle, edgecolor=c))
    if show_score:
        ax.text(x1, y1, '%.2f' % score,
                horizontalalignment='left', verticalalignment='bottom',
                backgroundcolor=c, color='k')
        
def vis_roidb(roidb, max_n_rois=None, vis_th=0.0, show_score=True):
    """
    Viasualize a trajectory. 
    Inputs:
    -------
    trajdb: list
        The trajectory, a list of dictionary.
    time_span: tuple of 2 or 3
        (start_frame, end_frame, step_size)
    rois: list
        Sampled RoIs of the traj, with the same size with traj.
    max_n_rois: int
        Maximum number of RoIs to plot.
    """
    # Create figure
    figsize = (10, 10)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    # Show the image
    img = cv2.imread(roidb['image'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    if max_n_rois is None:
        max_n_rois = roidb['boxes'].shape[0]
    keep_inds = np.where(roidb['gt_overlaps'] >= vis_th)[0]
    plt.title('Image: {}, n_rois: {}'.format(roidb['image'], min(len(keep_inds), max_n_rois)))
    
    # Plot the sampled rois
    rois_legend = 'ROI Scores:\n'
        
    for i in range(min(len(keep_inds), max_n_rois)):
        idx = keep_inds[i]
        box = roidb['boxes'][idx, :]

        score = roidb['gt_overlaps'][idx]
        if roidb.has_key('visible'):
            visible = roidb['visible'][idx]
        else:
            visible = True
        plot_box(ax, box, score, visible, show_score)
        rois_legend += '    %.2f\n' % score
    if show_score:
        plt.text(ax.get_xlim()[1], 0, rois_legend, va='top')
        
def vis_frame(cfg, ims, rois, scores,
            max_n_rois=None, vis_th=0.0, show_score=True, plot_th=0.0):
    """
    """
    
    # Create figure
    figsize = (10, 10)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    # Show the image
    img = ims[0] + np.array([[[102.9801, 115.9465, 122.7717]]])
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    if max_n_rois is None:
        max_n_rois = rois.shape[0]
    keep_inds = np.where(scores >= vis_th)[0]
    
    # Plot the sampled rois
    rois_legend = 'ROI Scores:\n'
        
    for i in range(min(len(keep_inds), max_n_rois)):
        idx = keep_inds[i]
        box = rois[idx, 1:]

        score = scores[idx]
        if score >= plot_th:
            plot_box(ax, box, score, True, show_score)
            rois_legend += '    %.2f\n' % score
    if show_score:
        plt.text(ax.get_xlim()[1], 0, rois_legend, va='top')
        

