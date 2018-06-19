import numpy as np
from scipy import misc
import os, time
import threading, queue
import classifier


# Produz imagens para posterior classificação
class produtor_de_imagem(threading.Thread):
    """
    Descrição da classe.
    """
    # Construtor de classe
    def __init__(self,filename_list,output_q):
        
        # Inicializador da classe base
        super(produtor_de_imagem,self).__init__()

        # Configura a comunicação da thread
        self.filename_list = filename_list
        self.output_q = output_q
        self.stop_request = threading.Event()

    def run(self):
        # Enquanto o evento stop_request não está setado,
        # tenta realizar sua tarefa
        while (not self.stop_request.isSet()):
            
            try:
                # Na aplicação, aqui aconteceria a captura da câmera
                for filename in self.filename_list:

                    # Aguarda 1 segundo, depois escreve a imagem na Queue
                    time.sleep(1)
                    img = misc.imread(filename)
                    img = img[:,:,0]
                    print(filename)
                    self.output_q.put(img) 
            
            except queue.Empty:
                continue

    def join(self, timeout=None):
        self.stop_request.set()
        super(produtor_de_imagem,self).join(timeout)
    

# Consome imagens para gerar classificação
#

class consumidor_de_imagem(threading.Thread):    
    """
    Desc. da classe.
    """
    # Construtor
    def __init__(self,classifier_obj,input_q,output_q):
        super(consumidor_de_imagem,self).__init__()

        # A Thread deve possuir um classificador para 
        # realizar sua tarefa
        self.classifier_obj = classifier_obj

        # Configura a comunicação da thread
        self.input_q = input_q
        self.output_q = output_q
        self.stop_request = threading.Event()
 


    # A thread de classificação pega uma
    # imagem carregada em np.array 
    def run(self):
        
        # Enquanto o evento stop_request não está setado,
        # tenta realizar sua tarefa
        while (not self.stop_request.isSet()):
                
            try:   
                image = self.input_q.get(True,0.05)
                label = self.classifier_obj.classify_img(image)
                print(label)
                self.output_q.put(label)
                                
            except queue.Empty:
                continue    

    # parada
    def join(self, timeout=None):
        self.stop_request.set()
        super(consumidor_de_imagem,self).join(timeout)


# Exibe as imagens sendo processadas pelo classificador
class visualizador_de_imagem(threading.Thread):
    
    # Construtor
    def __init__(self):
        super(visualizador_de_imagem,self).__init__()
        self.stop_request = threading.Event()
        
    # Tarefa da thread    
    def run(self):
        
        while (not self.stop_request.isSet()):
            try:
                pass
            except queue.Empty:
                continue
    
    # parada    
    def join(self,timeout=None):
        self.stop_request.set()
        super(consumidor,self).join(timeout)

def main():
    
    # Comunicação entre as threads
    image_q = queue.Queue()
    label_q = queue.Queue()
    
    # Cria thread pool
    thread_pool = []
    thread_pool.append(produtor(filename_list, image_q))
    thread_pool.append(consumidor(classificador,image_q, label_q))
    
    # Inicia threads
    for t in thread_pool:
        t.start()
        
    # Aguarda o fim da execução
    for t in thread_pool:
        t.join()
    
    
    
    
    
    
