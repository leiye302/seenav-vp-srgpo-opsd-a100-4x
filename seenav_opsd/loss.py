"""Add the pinned public SDAR loss to the unchanged native policy loss."""

from seenav_opsd.sdar_utils import compute_sdar_loss


def add_sdar_loss(policy_loss, student, teacher, response_mask, config):
    if config.sdar_loss_coef != 0.01 or config.sdar_gate_beta != 5.0 or config.loss_agg_mode != "token-mean":
        raise ValueError("Expected the frozen OPSD settings: coefficient .01, beta 5, token-mean.")
    if teacher.shape != student.shape or student.shape != response_mask.shape:
        raise ValueError("Teacher, Student and original response mask must align.")
    active = response_mask.bool()
    # Zero inactive positions before the stock loss to avoid masked NaNs.
    aux, metrics = compute_sdar_loss(
        student.masked_fill(~active, 0),
        teacher.detach().masked_fill(~active, 0),
        response_mask,
        gate_beta=config.sdar_gate_beta,
        loss_agg_mode=config.loss_agg_mode,
    )
    return policy_loss + config.sdar_loss_coef * aux, metrics
