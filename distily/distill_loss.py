from torch.nn import functional as F


def mse_loss(student_features, teacher_features):
    return F.mse_loss(student_features, teacher_features)


def kl_divergence_loss(student_features, teacher_features):
    student_log_prob = F.log_softmax(student_features, dim=-1)
    teacher_prob = F.softmax(teacher_features, dim=-1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")


def reverse_kl_divergence_loss(student_features, teacher_features):
    teacher_log_prob = F.log_softmax(teacher_features, dim=-1)
    student_prob = F.softmax(student_features, dim=-1)
    return F.kl_div(teacher_log_prob, student_prob, reduction="batchmean")


def cakld_loss(student_features, teacher_features, beta_prob=0.5):
    teacher_output_log_prob = F.log_softmax(teacher_features, dim=-1)
    student_output_soft = F.softmax(student_features, dim=-1)
    reverse_kl = F.kl_div(teacher_output_log_prob, student_output_soft, reduction="batchmean")

    student_output_log_prob = F.log_softmax(student_features, dim=-1)
    teacher_output_soft = F.softmax(teacher_features, dim=-1)
    forward_kl = F.kl_div(student_output_log_prob, teacher_output_soft, reduction="batchmean")

    kl_loss = beta_prob * reverse_kl + (1 - beta_prob) * forward_kl
    return kl_loss


def jsd_loss(student_features, teacher_features, beta_prob=0.5):
    student_prob = F.softmax(student_features, dim=-1)
    teacher_prob = F.softmax(teacher_features, dim=-1)

    c_prob = beta_prob * teacher_prob + (1 - beta_prob) * student_prob
    c_log_prob = c_prob.log()

    kl_loss_f = F.kl_div(c_log_prob, teacher_prob, reduction="batchmean")
    kl_loss_r = F.kl_div(c_log_prob, student_prob, reduction="batchmean")

    kl_loss = beta_prob * kl_loss_f + (1 - beta_prob) * kl_loss_r
    return kl_loss


LOSS_FUNCTIONS = {
    "mse": mse_loss,
    "kl": kl_divergence_loss,
    "reverse_kl": reverse_kl_divergence_loss,
    "cakld": cakld_loss,
    "jsd": jsd_loss
}
