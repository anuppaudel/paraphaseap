from django.shortcuts import render

from rest_framework.decorators import api_view
from rest_framework.response import Response
from .engine import *


model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase")
tokenizer = PegasusTokenizerFast.from_pretrained("tuner007/pegasus_paraphrase")


# Create your views here.
@api_view(['GET'])
def home(request):
    context ={
        'Response':'Setup success'
    }
    return Response(context)

@api_view(['POST'])
def pp(request):
    data = str(request.body)

    sentences = split_into_sentences(data)
    finresult =""
    for s in sentences:
        ppsentence = get_paraphrased_sentences(model, tokenizer, s, num_return_sequences=10, num_beams=10)
        finresult = finresult + " " + ppsentence[6]
    return Response(finresult)
