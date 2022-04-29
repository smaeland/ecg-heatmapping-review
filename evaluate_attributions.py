from enum import Enum
import numpy as np
import torch


class AttributeSign(Enum):
    positive = 1
    absolute_value = 2
    negative = 3
    all = 4


class AttrEvaluation:

    def __init__(
        self,
        data,
        attributions,
        channel_mode: str ='individual',
        attr_mode: str = 'absolute_value'
    ) -> None:
        
        assert len(data.shape) == 2
        assert attributions.shape == data.shape


        self.data = data
        if isinstance(data, torch.Tensor):
            self.data = data.detach().numpy()

        self.attr = attributions
        if isinstance(attributions, torch.Tensor):
            self.attr = attributions.detach().numpy()

        self.num_channels = data.shape[0]

        assert channel_mode in ['individual', 'merge']
        self.channel_mode = channel_mode
        if channel_mode == 'merge':
            self.merge_channels()

        self.attr_mode = AttributeSign[attr_mode]
        self.preprocess_attributes()


    def merge_channels(self):

        self.data = self.data[10]   # lead V5
        self.attrs = np.mean(self.attrs, axis=0)

        # Go from shape (600,) to (1, 600)
        self.data = np.expand_dims(self.data, axis=0)
        self.attrs = np.expand_dims(self.attrs, axis=0)
    

    def preprocess_attributes(self):
        
        _attr = self.attr

        if self.attr_mode == AttributeSign['positive']:
            _attr = (_attr > 0) * _attr
        elif self.attr_mode == AttributeSign['negative']:
            _attr = (_attr < 0) * _attr
        elif self.attr_mode == AttributeSign['absolute_value']:
            _attr = np.abs(_attr)

        self.attr = _attr

    def get_data_and_attrs(self):

        return self.data, self.attr



    def randomise_by_attributions(self):
        raise NotImplementedError


class RandomiseTopAttrs(AttrEvaluation):
    """
    Randomise the data at positions given by the top X percent attribution values
    """
    # TODO replace values by noise or add the noise to existing data?

    def __init__(
        self,
        data,
        attributions,
        channel_mode: str = 'individual',
        attr_mode: str = 'absolute_value',
        add_or_replace_noise: str = 'replace'
    ) -> None:
    
        super().__init__(data, attributions, channel_mode, attr_mode)

        assert add_or_replace_noise in ['add', 'replace', 'zero']
        self.add_or_replace_noise = add_or_replace_noise


    def randomise_by_attributions(self, top_percent=1, window_size=5, override_scale=None):

        assert window_size % 2 != 0, 'window_size should be odd'
        half_win_size = window_size // 2

        # Flatten data
        orig_data_shape = self.data.shape
        data = self.data.flatten()
        attr = self.attr.flatten()

        if override_scale is not None:
            stddev = override_scale
        else:
            stddev = np.std(data)*1.5

        # Get top X% attribution values 
        top_values = np.flip(np.sort(attr))
        n_elements = int(np.round(top_percent * 0.01 * len(attr)))
        top_values = top_values[:n_elements]

        # Get list of indices to randomize - for each attribution value, consider a window
        # Use a set to only keep unique values
        indices = set()
        for value in top_values:
            idx = np.where(attr == value)[0][0]
            indices.update(
                list(range(idx - half_win_size, idx + half_win_size + 1))
            )
        
        # Convert to ndarray and check bounds
        indices = np.array(list(indices))
        indices = indices[(indices >= 0) & (indices < len(data))]

        # Set random values
        if self.add_or_replace_noise == 'zero':
            data[indices] = np.zeros(shape=len(indices))

        else:
            rnd_vals = np.random.normal(size=len(indices), scale=stddev)
            if self.add_or_replace_noise == 'replace':
                data[indices] = rnd_vals
            else:
                data[indices] += rnd_vals

        # Convert back to correct shape
        self.data = np.reshape(data, newshape=orig_data_shape)

        print('Randomised {} values in data'.format(len(rnd_vals)))



class RandomiseIteratively(AttrEvaluation):
    """
    Randomise top attributions iteratively, like AOPC
    """

    def __init__(
        self,
        data,
        attributions,
        channel_mode: str = 'individual',
        attr_mode: str = 'absolute_value',
        add_or_replace_noise: str = 'replace'
    ) -> None:

        super().__init__(data, attributions, channel_mode, attr_mode)

        assert add_or_replace_noise in ['add', 'replace', 'zero']
        self.add_or_replace_noise = add_or_replace_noise


    def randomise_by_attributions(self, window_size, override_scale=None, verbose=False):
        """
        Generator yielding randomized data plus attributions
        """

        randomised_indices = set()
        iteration = 0

        assert window_size % 2 != 0, 'window_size should be odd'
        half_win_size = window_size // 2

        # Flatten data
        orig_data_shape = self.data.shape
        data = self.data.flatten()
        attr = self.attr.flatten()

        if override_scale is not None:
            stddev = override_scale
        else:
            #stddev = np.std(data)*2 #*1.5
            stddev = 0.2 * (np.max(data) - np.min(data))

        while True:

            iteration += 1

            # Find top attibution and get indices for entire window
            top_idx = np.where(attr == np.max(attr))[0][0]
            indices = list(range(top_idx - half_win_size, top_idx + half_win_size + 1))

            # Check bounds 
            inbounds = lambda x: x >= 0 and x < len(data)
            indices = list(filter(inbounds, indices))

            # Don't re-randomise indices covered by a previous iteration
            indices = set(indices) - randomised_indices
            #print('dbg: randomising at indices', indices)
            randomised_indices = randomised_indices.union(indices)

            # Randomise data and zero out the corresponding attributions
            indices = np.array(list(indices))
            if self.add_or_replace_noise == 'zero':
                rnd_vals = np.zeros(shape=len(indices))
            else:
                rnd_vals = np.random.normal(size=len(indices), scale=stddev)
            
            if self.add_or_replace_noise == 'add':
                data[indices] += rnd_vals
            else:
                data[indices] = rnd_vals
            
            attr[indices] = np.zeros(shape=len(indices))

            # Convert back to correct shape and yield results
            out_data = np.reshape(data, newshape=orig_data_shape)
            out_attr = np.reshape(attr, newshape=orig_data_shape)

            if verbose:
                print('Iteration {}: {}/{} ({:.2f}%) data points randomised'.format(
                    iteration,
                    len(randomised_indices),
                    len(data),
                    len(randomised_indices) / len(data) * 100.0
                ))

            yield out_data, out_attr





    




class RandomiseByValue(AttrEvaluation):
    """
    Add random noise at all locations, where the scale of the noise is given by the attribution value
    """

    def __init__(
        self,
        data,
        attributions,
        channel_mode: str = 'individual',
        attr_mode: str = 'absolute_value'
    ) -> None:

        assert attr_mode in ['absolute_value', 'positive'], 'Need positive attributions for randomisation'

        super().__init__(data, attributions, channel_mode, attr_mode)

    
    def normalise_attrs(self, outlier_percentage=2):

        values = self.attr
        sorted_vals = np.sort(values.flatten())
        cum_sums = np.cumsum(sorted_vals)
        threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * (100 - outlier_percentage))[0][0]
        scale_factor = sorted_vals[threshold_id]

        attr_norm = values / scale_factor

        return np.clip(attr_norm, -1, 1)

    

    def randomise_by_attributions(self, lower_cutoff=0.3, overall_scale_override=None):

        # Normalise attributions
        attr = self.normalise_attrs()

        # Flatten data
        orig_data_shape = self.data.shape
        data = self.data.flatten()
        attr = attr.flatten()

        data_out = np.array(data)
        avg_scale = 0

        stddev = np.std(data)*2
        if overall_scale_override is not None:
            stddev = overall_scale_override


        # Add noise for each point
        for i in range(len(data)):
            
            noise = 0
            if attr[i] > lower_cutoff:
                scale = attr[i] * stddev
                avg_scale += scale
                noise = np.random.normal(scale=scale)

            data_out[i] = data[i] + noise

        # Convert back to correct shape
        self.data = np.reshape(data_out, newshape=orig_data_shape)

        avg_scale /= len(data)
        print('Added noise with avg scale = {}'.format(avg_scale))




        
        




        




        
        
