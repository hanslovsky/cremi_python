#!/usr/bin/env python3

from cremi import Volume
from cremi.evaluation import border_mask, rand, voi
import json
import numpy as np
import os
import shutil
import sys
import z5py

def run_evaluation():

    experiment_dir    = '/nrs/saalfeld/hanslovskyp/experiments/quasi-isotropic-predictions/affinities-glia/neuron_ids-unlabeled-unmask-background'
    container_pattern = f'{experiment_dir}/predictions/CREMI/sample_%(sample)s.n5'
    group_pattern     = '/volumes/predictions/neuron_ids-unlabeled-unmask-background' + \
                        '/%(setup)d/%(iteration)d/mutex-watershed%(threshold)s'

    ground_truth_container_pattern = '/groups/saalfeld/home/hanslovskyp/data/from-arlo/interpolated-combined/sample_%(sample)s.n5'
    ground_truth_dataset           = '/volumes/labels/neuron_ids_noglia-downsampled'
    mask_dataset                   = '/volumes/labels/mask-downsampled'
    mask_dataset_75                = '/volumes/labels/mask-downsampled-75%-y'
    glia_dataset                   = '/volumes/labels/glia-downsampled'
    glia_prediction_pattern        = '/volumes/predictions/neuron_ids-unlabeled-unmask-background/%(setup)d/500000/glia'
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', choices=tuple('ABC012'), required=True)
    parser.add_argument('--setup', choices=tuple(range(0, 6)), type=int, required=True)
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--threshold', type=str, choices=('0.1', '0.3', '0.5', '0.7', '0.9'), required=False)
    parser.add_argument('--border-threshold', type=float, required=False)
    parser.add_argument('--log-data', type=str, default=None, required=False)
    parser.add_argument('--remove-log-dir', action='store_true')
    parser.add_argument('--ignore-glia', type=float, default=None)
    parser.add_argument('--ignore-training-mask', action='store_true')
    parser.add_argument('--overwrite-existing', action='store_true')

    args = parser.parse_args()

    container               = container_pattern % dict(sample=args.sample)
    ground_truth_container  = ground_truth_container_pattern % dict(sample=args.sample)
    group                   = group_pattern % dict(
        setup           = args.setup,
        threshold      = '' if args.threshold is None else ('-threshold=%s' % args.threshold),
        iteration       = args.iteration)
    dataset                 = f'{group}-merged-with-glia'
    glia_prediction_dataset = glia_prediction_pattern % dict(setup=args.setup)

    with z5py.File(container, 'r') as f:
        if 'completedSuccessfully' not in f[dataset].attrs:
            print('No `completedSuccessfully\' tag in group! Will not evaluate')
            raise Exception('No `completedSuccessfully\' tag in group! Will not evaluate')
        elif not f[dataset].attrs['completedSuccessfully']:
            print('`completedSuccessfully\' tag in group is false! Will not evaluate')
            raise Exception('`completedSuccessfully\' tag in group is false! Will not evaluate')
        # print(f[group].attrs['completedSuccessfully'])

    evaluation_state     = dict(was_successful=False)
    evaluation_base_path = 'evaluation_border-threshold=%s' % str(args.border_threshold)
    evaluation_base_path = f'{evaluation_base_path}_ignore-training-mask' if args.ignore_training_mask else evaluation_base_path
    evaluation_base_path = f'{evaluation_base_path}_ignore-glia=%s' % str(args.ignore_glia) if args.ignore_glia else evaluation_base_path
    evaluation_file_name = f'{container}{group}_{evaluation_base_path}.json'

    try:
        with open(evaluation_file_name, 'r') as f:
            existing_evaluation_state = json.load(f)
            # print(evaluation_state)
            if 'was_successful' in existing_evaluation_state and existing_evaluation_state['was_successful'] and not args.overwrite_existing:
                print('Already evaluated this! Use the --overwrite-existing flag to re-run evaluation.')
                return 1
    except FileNotFoundError:
        # no evaluation yet, do not do anything
        pass

    with open(evaluation_file_name, 'w') as f:
        json.dump(evaluation_state, f)

    print(container)
    print(ground_truth_container)
    print(dataset)

    with z5py.File(ground_truth_container, 'r') as f:

        resolution  = np.array(f[mask_dataset].attrs['resolution'])
        offset      = np.array(f[mask_dataset].attrs['offset'])
        shape       = f[mask_dataset].shape
        mask        = Volume(np.require(f[mask_dataset][:shape[0], :shape[1], :shape[2]], dtype=np.int64), resolution, offset)

        resolution  = np.array(f[mask_dataset_75].attrs['resolution'])
        offset      = np.array(f[mask_dataset_75].attrs['offset'])
        shape       = f[mask_dataset_75].shape
        mask_75     = Volume(np.require(f[mask_dataset_75][:shape[0], :shape[1], :shape[2]], dtype=np.int64), resolution, offset)

        resolution   = np.array(f[ground_truth_dataset].attrs['resolution'])
        offset       = np.array(f[ground_truth_dataset].attrs['offset'])
        shape        = f[ground_truth_dataset].shape
        ground_truth = Volume(np.require(f[ground_truth_dataset][:shape[0], :shape[1], :shape[2]], dtype=np.int64), resolution, offset)
        ground_truth.data += 1

        resolution   = np.array(f[ground_truth_dataset].attrs['resolution'])
        offset       = np.array(f[ground_truth_dataset].attrs['offset'])
        shape        = f[glia_dataset].shape
        assert np.all(resolution == ground_truth.resolution)
        assert np.all(offset == ground_truth.offset)
        assert shape == ground_truth.data.shape
        glia         = np.array(f[glia_dataset][:shape[0], :shape[1], :shape[2]])
        ground_truth.data[glia == 1] = 1

        ignore_mask = mask.data == 0
        if  args.ignore_training_mask:
            resolution  = np.array(f[mask_dataset_75].attrs['resolution'])
            offset      = np.array(f[mask_dataset_75].attrs['offset'])
            shape       = f[mask_dataset_75].shape
            assert np.all(resolution == mask.resolution)
            assert np.all(offset == mask.offset)
            assert shape == mask.data.shape
            mask_75     = Volume(np.require(f[mask_dataset_75][:shape[0], :shape[1], :shape[2]], dtype=np.int64), resolution, offset)
            ignore_mask[mask_75.data == 1] = 1
            print(np.sum(ignore_mask), np.sum(mask_75.data), np.sum(mask.data == 0))

    assert np.all(mask.resolution == ground_truth.resolution)
    assert np.all(mask.offset == ground_truth.offset)

    with z5py.File(container, 'r') as f:

        resolution = np.array(f[dataset].attrs['resolution'][::-1])
        offset     = np.array(f[dataset].attrs['offset'][::-1])

        assert np.all(resolution == ground_truth.resolution), '%s != %s' % (resolution, ground_truth.resolution)
        offset = np.floor(ground_truth.offset / resolution).astype(np.int64)
        shape  = ground_truth.data.shape
        slicing = tuple(slice(o, o + s) for o, s in zip(offset, shape))
        prediction = Volume(np.require(f[dataset][slicing], dtype=np.int64), resolution, offset)
        slicing = tuple(slice(s) for s in shape)

        if args.ignore_glia is not None:
            glia_prediction = Volume(np.array(f[glia_prediction_dataset][slicing]), resolution, offset)
            ignore_mask[glia_prediction.data > args.ignore_glia] = True

    # cremi eval cannot do uint64 because of this weirdness:
    # >>> type(np.uint64(3) + 1)
    # <class 'numpy.float64'>
    ground_truth.data[ground_truth.data < 0] = 0
    prediction.data[prediction.data < 0] = 0
    # ground_truth.data[ground_truth.data == 0] = -1
    # ground_truth.data[ground_truth.data > np.uint64(-100)] = 0
    # ground_truth.data[ground_truth.data == 0] = np.uint64(-1)

    # training_mask = mask.data == 1
    # ground_truth.data[training_mask] = 0
    # prediction.data[training_mask] = 0

    distances, thresholded = None, None
    if args.border_threshold is not None:
        print("Calculating border mask for ground truth with threshold=%f" % args.border_threshold)
        distances, thresholded = border_mask.create_border_mask_3d(
            image    = ground_truth.data,
            max_dist = args.border_threshold,
            weights  = ground_truth.resolution)
        # border_mask.create_border_mask(
        #     ground_truth.data,
        #     gt,
        #     float(args.border_threshold) / ground_truth.resolution[1],
        #     0)
        # thresholded is within threshold to boundary --> ignore
        ignore_mask[thresholded] = True

    ground_truth.data[ignore_mask] = 0

    if args.log_data is not None:
        print('Logging data into %s' % args.log_data)
        if (args.remove_log_dir):
            shutil.rmtree(args.log_data, ignore_errors=True)
        os.makedirs(args.log_data, exist_ok=True)
        with z5py.File(args.log_data, 'w') as f:
            ds1 = f.create_dataset('/gt', data=ground_truth.data.astype(np.uint64), chunks=(64,64,64))
            ds2 = f.create_dataset('/pred', data=prediction.data.astype(np.uint64), chunks=(64,64,64))
            ds3 = f.create_dataset('/mask', data=mask.data, chunks=(64,64,64))
            ds4 = f.create_dataset('/dist', data=distances, chunks=(64,64,64)) if distances is not None else None
            ds5 = f.create_dataset('/thresh', data=thresholded.astype(np.uint8), chunks=(64,64,64)) if thresholded is not None else None
            ds6 = f.create_dataset('/ignore', data=ignore_mask.astype(np.uint8), chunks=(64,64,64))

            for ds in (ds1, ds2, ds3, ds4, ds5, ds6):
                if ds is not None:
                    ds.attrs['resolution'] = tuple(ground_truth.resolution)
                    ds.attrs['offset']     = tuple(ground_truth.offset)

            if ds4 is not None:
                ds4.attrs['value_range'] = (np.min(distances), np.max(distances))
            if ds5 is not None:
                ds5.attrs['value_range'] = (0.0, 1.0)
            ds6.attrs['value_range'] = (0.0, 1.0)

    print("calculating statistics")
    # voi_split, voi_merge = voi(ground_truth.data, ground_truth.data, ignore_groundtruth=[0])
    voi_split, voi_merge = voi(prediction.data, ground_truth.data, ignore_groundtruth=[0])
    print('voi_split', voi_split)
    print('voi_merge', voi_merge)
    # adapted_rand, prec, rec = rand.adapted_rand(ground_truth.data, ground_truth.data, all_stats=True)
    adapted_rand, prec, rec = rand.adapted_rand(prediction.data, ground_truth.data, all_stats=True)
    print('adapted_rand', adapted_rand, prec, rec)

    evaluation_state['was_successful'] = True
    evaluation_state['scores'] = dict(
        voi_split    = voi_split,
        voi_merge    = voi_merge,
        adapted_rand = adapted_rand,
        precision    = prec,
        recall       = rec)
    with open(evaluation_file_name, 'w') as f:
        json.dump(evaluation_state, f)

    return 0

if __name__ == '__main__':
    sys.exit(run_evaluation())

    
