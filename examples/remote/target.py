sing_octo_target = \
"""
target:
  service: aisc
  name: msroctovc

environment:
  image: amlt-sing/pytorch-1.8.0
  setup:
  - . ./examples/remote/setup.sh"""

sing_research_target = \
"""
target:
  service: aisc
  name: msrresrchvc

environment:
  image: amlt-sing/pytorch-1.8.0
  setup:
  - . ./examples/remote/setup.sh"""

itp_rr1_target = \
"""
target:
  service: amlk8s
  name: itplabrr1cl1
  vc: resrchvc

environment:
  image: v-xudongwang/pytorch:taoky
  username: resrchvc4cr
  registry: resrchvc4cr.azurecr.io
  setup:
  - . ./examples/remote/setup.sh"""

itp_ads_v100_eus1_target = \
"""
target:
  service: amlk8s
  name: v100-8x-eus-1
  vc: ads

environment:
  image: v-xudongwang/pytorch:taoky
  username: resrchvc4cr
  registry: resrchvc4cr.azurecr.io
  setup:
  - . ./examples/remote/setup.sh"""

itp_ads_gpt3_target = \
"""
target:
  service: amlk8s
  name: v100-8x-wus2
  vc: Ads-GPT3

environment:
  image: v-xudongwang/pytorch:taoky
  username: resrchvc4cr
  registry: resrchvc4cr.azurecr.io
  setup:
  - . ./examples/remote/setup.sh"""

itp_ads_a100_target = \
"""
target:
  service: amlk8s
  name: a100-8x-wus2
  vc: ads

environment:
  image: v-xudongwang/pytorch:cddlyf_a100
  username: resrchvc4cr
  registry: resrchvc4cr.azurecr.io
  setup:
  - . ./examples/remote/setup.sh"""

target_dict = dict(
    sing_octo=sing_octo_target,
    sing_research=sing_research_target,
    itp_rr1=itp_rr1_target,
    itp_ads_v100_eus1=itp_ads_v100_eus1_target,
    itp_ads_gpt3=itp_ads_gpt3_target,
    itp_ads_a100=itp_ads_a100_target
)