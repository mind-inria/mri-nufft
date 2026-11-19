"""Batch, coil and SENSE configurations of the operators we want to test."""


class CasesBatch:
    """Operator configurations.

    Each case returns ``(n_batchs, n_coils, n_trans, sense)``.
    """

    def case_single(self):
        """One image, one coil."""
        return 1, 1, 1, False

    def case_batch(self):
        """Several images, one coil."""
        return 3, 1, 1, False

    def case_sense(self):
        """One image, several coils combined with SENSE."""
        return 1, 4, 1, True

    def case_coils_ntrans(self):
        """One image, several coils, transformed two at a time."""
        return 1, 4, 2, False

    def case_batch_sense(self):
        """Several images, several coils combined with SENSE."""
        return 3, 2, 1, True

    def case_batch_sense_ntrans(self):
        """Several images, several coils with SENSE, transformed two at a time."""
        return 3, 4, 2, True

    def case_batch_coils_ntrans(self):
        """Several images, several coils, transformed two at a time."""
        return 3, 4, 2, False
