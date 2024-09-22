# Desenvolvimento Front-End com Python (com Streamlit)
### TP3

## Instruções:

> Este teste de performance contém 12 itens que devem ser realizados sequencialmente. Cada item está relacionado a uma competência específica no desenvolvimento de aplicações com Streamlit. Utilize os dados provenientes do portal Data.Rio, seção turismo, e crie uma interface atraente para exibi-los. 

[Data Rio](https://www.data.rio/search?groupIds=729990e9fbc04c6ebf81715ab438cae8)

🟢 Sinal verde: Os alunos são incentivados a explorar o uso de ferramentas baseadas em IA para concluir as tarefas. Todas as fontes, incluindo ferramentas de IA, devem ser devidamente citadas. O uso de IA sem a devida citação será considerado má conduta acadêmica e estará sujeito à aplicação do código disciplinar. Observe que os resultados da IA podem ser tendenciosos e imprecisos. É sua responsabilidade garantir que as informações que você usa da IA sejam precisas. Aprender como usar ferramentas baseadas em IA de maneira cuidadosa e estratégica contribui para o desenvolvimento das habilidades, refinamento de seu trabalho e prepara o aluno para sua futura carreira.

``` Foi utilizado IA para auxílio na codificação (github copilot) ```

### Resolva os seguintes itens: 

1. Explicação do Objetivo e Motivação:

    Explique o objetivo do dashboard que você está desenvolvendo e a motivação por trás da escolha dos dados e funcionalidades que serão implementados.

    ```Dashboard focado na analise do fluxo de turistas estrangeiros vindos por via area ao Rio de Janeiro, no intervalo de 2006 até 2019. Entender origem dos turistas, frequência ao longo dos meses e anos da série.```

2. Realizar Upload de Arquivo CSV:

    Crie uma interface em Streamlit que permita ao usuário fazer o upload de um arquivo CSV contendo dados de turismo do portal Data.Rio.
    
    ```Coloquei a funcionalidade de carregar Excel, pois os dados que escolhi vem nesse formato.```
 
3. Filtro de Dados e Seleção:

    Implemente seletores (radio, checkbox, dropdowns) na interface que permitam ao usuário filtrar os dados carregados e selecionar as colunas ou linhas que deseja visualizar.

    ```No SideBar e Container de Dados.```


4. Desenvolver Serviço de Download de Arquivos:

    Implemente um serviço que permita ao usuário fazer o download dos dados filtrados em formato CSV diretamente pela interface da aplicação.

5. Utilizar Barra de Progresso e Spinners:

    Adicione uma barra de progresso e um spinner para indicar o carregamento dos dados enquanto o arquivo CSV é processado e exibido na interface.

    ```No processo de carga de um excel podemos ver ambos.```

6. Utilizar Color Picker:

    Adicione um color picker à interface que permita ao usuário personalizar a cor de fundo do painel e das fontes exibidas na aplicação.

    ``` no SideBar ```

7. Utilizar Funcionalidade de Cache:

    Utilize a funcionalidade de cache do Streamlit para armazenar os dados carregados de grandes arquivos CSV, evitando a necessidade de recarregá-los a cada nova interação.

    ``` na carga de dados ```

8. Persistir Dados Usando Session State:

    Implemente a persistência de dados na aplicação utilizando Session State para manter as preferências do usuário (como filtros e seleções) durante a navegação.

    ``` nas preferencias do usuario e filtros ```


9. Criar Visualizações de Dados - Tabelas:

    Crie uma tabela interativa que exiba os dados carregados e permita ao usuário ordenar e filtrar as colunas diretamente pela interface.
 

10. Criar Visualizações de Dados - Gráficos Simples:

    Desenvolva gráficos simples (como barras, linhas, e pie charts) para visualização dos dados carregados, utilizando o Streamlit.

11. Criar Visualizações de Dados - Gráficos Avançados:

    Adicione gráficos mais avançados (como histograma ou scatter plot) para fornecer insights mais profundos sobre os dados.

12. Exibir Métricas Básicas:

    Implemente a exibição de métricas básicas (como contagem de registros, médias, somas) diretamente na interface para fornecer um resumo rápido dos dados carregados.


> Ao concluir, certifique-se de que a interface criada seja funcional, intuitiva e esteticamente agradável, proporcionando uma boa experiência ao usuário. Cada item deve ser implementado de forma a complementar a aplicação final, garantindo que todas as competências sejam avaliadas.

O código final deve ser entregue no github.

Assim que terminar, salve seu trabalho em PDF (com o endereço do Github) nomeando o arquivo conforme a regra “nome_sobrenome_DR1_TP3.PDF” e poste como resposta a este TP.