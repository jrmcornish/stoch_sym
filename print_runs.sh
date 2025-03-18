#!/bin/bash

set -o errexit
set -o nounset

main() {
	print_covariance_estimation
	print_linear_regression
	print_matrix_exponentiation
	print_matrix_inversion
}

print_covariance_estimation() {
	print_kim_et_al          --dataset=cov --group=orth --input-action=lmul --output-action=conj
	print_our_method         --dataset=cov --group=orth --input-action=lmul --output-action=conj
	print_unsymmetrised_mlp  --dataset=cov --group=orth --input-action=lmul --output-action=conj
	print_emlp               --dataset=cov --group=orth --input-action=lmul --output-action=conj
	print_haar_as_base_case  --dataset=cov --group=orth --input-action=lmul --output-action=conj
}

print_linear_regression() {
	print_kim_et_al          --dataset=linreg --group=orth --input-action=lmul-triv --output-action=lmul
	print_our_method         --dataset=linreg --group=orth --input-action=lmul-triv --output-action=lmul
	print_unsymmetrised_mlp  --dataset=linreg --group=orth --input-action=lmul-triv --output-action=lmul
	print_emlp               --dataset=linreg --group=orth --input-action=lmul-triv --output-action=lmul
	print_haar_as_base_case  --dataset=linreg --group=orth --input-action=lmul-triv --output-action=lmul
}

print_matrix_exponentiation() {
	print_kim_et_al          --dataset=expm --group=orth --input-action=conj --output-action=conj
	print_our_method         --dataset=expm --group=orth --input-action=conj --output-action=conj
	print_unsymmetrised_mlp  --dataset=expm --group=orth --input-action=conj --output-action=conj
	print_emlp               --dataset=expm --group=orth --input-action=conj --output-action=conj
	print_haar_as_base_case  --dataset=expm --group=orth --input-action=conj --output-action=conj
}

print_matrix_inversion() {
	print_kim_et_al          --dataset=inv --group=orth --input-action=lmul --output-action=rmul
	print_our_method         --dataset=inv --group=orth2 --input-action=lmul-rmul --output-action=rmul-lmul
	print_unsymmetrised_mlp  --dataset=inv --group=orth2 --input-action=lmul-rmul --output-action=rmul-lmul
	print_emlp               --dataset=inv --group=orth --input-action=lmul --output-action=rmul
	print_haar_as_base_case  --dataset=inv --group=orth2 --input-action=lmul-rmul --output-action=rmul-lmul
}

print_kim_et_al() {
	print "$@" --backbone mlp --gamma emlp --config "base_hidden_channels=[250,250]" --config "gamma_hidden_channels=[250]"
}

print_our_method() {
	print "$@" --backbone mlp --gamma mlp-haar --config "base_hidden_channels=[250,250]" --config "gamma_hidden_channels=[250]"
}

print_haar_as_base_case() {
	print "$@" --backbone mlp --gamma haar --config "base_hidden_channels=[250,250,250]"
}

print_unsymmetrised_mlp() {
	print "$@" --backbone mlp --gamma none --config "base_hidden_channels=[250,250,250]"
}

print_emlp() {
	print "$@" --backbone emlp --gamma none --config "base_hidden_channels=[250,250,250]"
}

print() {
	for dim in 4 8 16 32; do
		echo --seed $RANDOM --config "dim=$dim" "$@"
	done
}

main