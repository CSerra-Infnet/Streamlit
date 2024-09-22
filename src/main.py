import streamlit as st
import glob
import pandas as pd
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
import seaborn as sns
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import pydeck as pdk
import geopandas as gpd
from functools import reduce
from deep_translator import MyMemoryTranslator 
from itertools import zip_longest

st.set_page_config(page_title="TP3", page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)


dataframe_dict = {}
dataframe_combined = None
last_continent = ''

# https://stackoverflow.com/questions/5389507/iterating-over-every-two-elements-in-a-list
def grouped(iterable, n) -> tuple:
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return zip_longest(*[iter(iterable)]*n)

#https://blog.streamlit.io/auto-generate-a-dataframe-filtering-ui-in-streamlit-with-filter_dataframe/
def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns
    
    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Filtrar Dados")

    if not modify:
        return df

    df = df.copy()
    # para poder filtrar pelos indices
    df.reset_index(inplace=True)
    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filtrar em", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 20:
                user_cat_input = right.multiselect(
                    f"Valores para {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Valores para {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Valores para {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring ou regex em {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df

def row_type(row) -> int:
    """
    Função que verifica se a linha é um continente
    """
    if row.local in ['África', 'América Central', 'América do Sul', 'América do Norte', 'Ásia', 'Europa', 'Oriente Médio', 'Oceania']:
        return 1
    if row.local == "Total":
        return 0
    if row.local == "Países não especificados":
        return 3
    return 2

def get_continent(row) -> str:
    """
    Função que obtem o continente da linha olhando o ultimo continente encontrado
    """
    if row.tipo == 1:
        st.session_state.last_continent = row.local
        return row.local
    if row.tipo == 0:
        return '-'
    if row.tipo == 2:
        return st.session_state.last_continent
    if row.tipo == 3:
        return "Zona não especificada"

def convert_to_datetime(row) -> pd.Timestamp | None:
    """
    Função que converte a string para datetime
    """
    ano, mes = row.split('_')
    meses = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho', 'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']
    try:
        mes = meses.index(mes) + 1        
        return pd.to_datetime(f'{ano}-{mes}', format='%Y-%m', errors='coerce')
    except:
        return None


@st.cache_data
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

@st.cache_data
def load_country_data() -> pd.DataFrame:
    """
    Load data from file path and load into a DataFrame dictionary
    """
    # data = pd.read_csv('https://raw.githubusercontent.com/google/dspl/master/samples/google/canonical/countries.csv')
    
    # data['name'] = data['name'].apply(lambda x: MyMemoryTranslator(source='en-US', target='pt-BR').translate(text=x))
    
    data = pd.read_csv('data/01_raw/geoData.csv')
    
    return data    

@st.cache_data
def load_data() -> dict:
    """
    Load data from file path and load into a DataFrame dictionary
    """
    all_sheets = {}
    for f in glob.glob('data/01_raw/*.xls'):
        parsed_data = parse_excel_file(f)
        all_sheets.update(parsed_data)
    return all_sheets

@st.cache_data       
def combine_data(data: dict) -> pd.DataFrame|None:
    """
    Combine all DataFrames into a single DataFrame
    """
    dataframe_combined = None
    with st.spinner('Combinando planilhas...'):               
        dataframe_combined = reduce(lambda  left,right: pd.merge(left,right, left_index=True, right_index=True,  how='outer'), data.values()).fillna(0)
        st.success('Planilhas combinadas com sucesso!')
        
    return dataframe_combined

                
def upload_csv_file() -> None:
    st.session_state["upload_csv"] = True
            
def upload_excel_file() -> None:
    st.session_state["upload_excel"] = True

def checkbox_container(data):
    cols = st.columns(2)
    if cols[0].button('Sel. Todos'):
        for i in data:
            st.session_state['dynamic_checkbox_' + i] = True
        #st.experimental_rerun()
    if cols[1].button('Rem. Todos'):
        for i in data:
            st.session_state['dynamic_checkbox_' + i] = False
        #st.experimental_rerun()

    ckcols = st.columns(4)
    for ix, i in enumerate(data):
        ckcols[ix % 4].checkbox(i, key='dynamic_checkbox_' + i, value=st.session_state.get('dynamic_checkbox_' + i, False))

def get_selected_checkboxes() -> list:
    return [i.replace('dynamic_checkbox_','') for i in st.session_state.keys() if i.startswith('dynamic_checkbox_') and st.session_state[i]]

def parse_excel_file(uploaded_excel) -> dict:
    progress_text = "Carregando planilhas, aguarde..."
    my_bar = st.progress(0, text=progress_text)
    dataframes = {}
    xl = pd.ExcelFile(uploaded_excel)                
    for sheet_name in xl.sheet_names:
        if sheet_name.isnumeric(): # carrega somente as planilhas que são anos, na ordem original
            dataframes[sheet_name] = xl.parse(sheet_name, header=5, index_col=None)
            # renomea coluna 0 para local
            dataframes[sheet_name].rename(columns={dataframes[sheet_name].columns[0]: 'local'}, inplace=True)
            dataframes[sheet_name]["local"] = dataframes[sheet_name]["local"].str.strip()
            #dataframes[sheet_name].index.rename('local', inplace=True)
            #dataframes[sheet_name].index = dataframes[sheet_name].index.str.strip()
            dataframes[sheet_name].columns = [sheet_name + "_" + col if col != 'local' else 'local' for col in dataframes[sheet_name].columns.str.strip() ]
            dataframes[sheet_name].dropna(inplace=True)
            dataframes[sheet_name].insert(0, 'tipo', '')
            dataframes[sheet_name].insert(0, 'continente', '')

            dataframes[sheet_name].reset_index(inplace=True)
           
            # loop para preencher a coluna continente, se não tiver ordenado por continente não funciona
            for index, row in dataframes[sheet_name].iterrows():
                dataframes[sheet_name].at[index, 'tipo'] = row_type(row)
                dataframes[sheet_name].at[index, 'continente'] = get_continent(dataframes[sheet_name].loc[index])
            
           
            dataframes[sheet_name].drop(columns=['index'], inplace=True)
            #st.write(dataframes[sheet_name].head(25))
            # define o index como continente e o index original 
            #dataframes[sheet_name].reset_index(drop=True, inplace=True)
            dataframes[sheet_name].set_index(['tipo', 'continente', 'local'], drop=True, inplace=True)
            # ordena o index
            #dataframes[sheet_name].sort_index(inplace=True)
            # converte colunas para nunmerico
            dataframes[sheet_name] = dataframes[sheet_name].apply(pd.to_numeric, errors='coerce')
        my_bar.progress((list(xl.sheet_names).index(sheet_name) + 1) / len(xl.sheet_names))
    my_bar.empty()
    st.write("Dados Carregados com Sucesso!") 
    return dataframes

def limpa_filtro() -> None:
    for i in st.session_state.keys():
        if i.startswith('dynamic_checkbox_'):
            del st.session_state[i]


def Dashboard() -> None:
    """
    Main function to create the Dashboard
    """
    if 'last_continent' not in st.session_state:
        st.session_state.last_continent = ''
        
    dataframe_dict = load_data()
    st.session_state.dataframe_dict = dataframe_dict
    # for k in dataframe_dict: break
    # st.write(dataframe_dict[k].head(20))
    
    # return None
    dataframe_combined = combine_data(dataframe_dict)
    st.session_state.dataframe_combined = dataframe_combined
    
    geo_data = load_country_data()
    #st.write(geo_data)
    
    if 'userPreferences' in st.session_state:
        st.markdown(
            f"""
            <style>
                * {{
                     color: {st.session_state.userPreferences['fontColor']} !important;
                }}
                [data-testid="stApp"]{{
                    background-color: {st.session_state.userPreferences['bgColor']} !important;
                }}
                [data-testid="stHeader"]{{
                    background-color: {st.session_state.userPreferences['bgColor']} !important;
                }}
                [data-testid="stSidebar"]{{
                    background-color: {st.session_state.userPreferences['sideColor']} !important;
                }}
            </style>
            """,
            unsafe_allow_html=True,
        )

    

    st.title("Dashboard Viagens de Estrangeiros ao Rio de Janeiro por Via Aerea (2006-2019) - TP3")

    if 'upload_excel' in st.session_state:
        uploaded_excel = st.file_uploader("Escolha um arquivo Excel", type=["xls", "xlsx"])
        if uploaded_excel is not None:
            try:
                parsed_data = parse_excel_file(uploaded_excel)
                st.session_state.dataframe_dict.update(parsed_data)
                dataframe_dict = st.session_state.dataframe_dict               
                # dataframe_combined = combine_data(dataframe_dict)
                # st.session_state.dataframe_combined = dataframe_combined
                
            except Exception as e:
                st.error(e)
            finally:
                del st.session_state["upload_excel"]
                # st.experimental_rerun()

    
    if 'upload_csv' in st.session_state:
        uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.dataframe_dict[uploaded_file.name] = data
                dataframe_dict = st.session_state.dataframe_dict
                with st.popover("Dados Carregados com Sucesso!"):
                    st.write(dataframe_dict[uploaded_file.name].head())
            except Exception as e:
                st.error(e)
            finally:
                del st.session_state["upload_csv"]
                # st.experimental_rerun()

    selected = None    
 
    # Menu lateral com carga de Arquivos e Seleção de Planilha
    with st.sidebar:
        with st.container():            
            st.header("Preferências")
            # color picker para cor de background
            bgcolor = st.color_picker("Escolha a cor de fundo", '#f0f0f0')
            
             # color picker para cor de background
            sidecolor = st.color_picker("Escolha a cor do menu lateral", '#CCCCCC')
            # slider para escolher o tamanho da fonte
            fontColor = st.color_picker("Escolha a cor da fonte", '#000000')
            # slider para escolher o tamanho da fonte
            fontSize = None # fontSize = st.slider("Escolha o tamanho da fonte", 10, 50, 20)
            userPreferences = {'bgColor': bgcolor, 'sideColor':sidecolor, 'fontColor': fontColor, 'fontSize': fontSize}
            st.session_state.userPreferences = userPreferences
            
            if len(dataframe_dict) > 0:
                full_list = list(dataframe_dict.keys())
                full_list.append('Combinados')
                st.write("### Selecione a fonte de dados")
                selected = st.selectbox("planilha", full_list, on_change=limpa_filtro)
            
            st.button(label="Carregar Excel", on_click=upload_excel_file)
            st.button(label="Carregar CSV", on_click=upload_csv_file)
            
            
            # csv =  convert_df(geo_data)
            
            # st.download_button(
            # "Download GeoData CSV",
            # csv,
            # f'geoData.csv',
            # "text/csv"
            # )
            
    # if dataframe_combined != None:   
        
    #     combined_continent = dataframe_combined[dataframe_combined['tipo'] ==  1]
    #     combined_countries = dataframe_combined[dataframe_combined['tipo'] == 2]
    #     combined_countries = combined_countries.drop(columns=['tipo'])
        
        
    if selected:
        # seleciona via checkbox quais colunas do dataframe serão exibidas
        
        st.header("Filtros")
        with st.expander("Selecione as colunas a serem exibidas"):
            checkbox_container(dataframe_combined.columns if selected == 'Combinados' else  dataframe_dict[selected].columns)
        
        selected_columns = get_selected_checkboxes()
        if not selected_columns:
            selected_columns = dataframe_combined.columns if selected == 'Combinados' else  dataframe_dict[selected].columns
            
        dataframe_show = dataframe_combined[selected_columns] if selected == 'Combinados' else dataframe_dict[selected][selected_columns]
        # if selected_columns:
        #     dataframe_show = dataframe_show[[col for col in dataframe_show.columns if col in selected_columns]]
        filtered_df = filter_dataframe(dataframe_show)
        filtered_df.reset_index(inplace=True)
        st.write(filtered_df)
        
        csv =  convert_df(filtered_df)
        
        st.download_button(
        "Download CSV",
        csv,
        f'{selected}.csv',
        "text/csv"
        )
        
        all_continents = filtered_df[filtered_df['tipo']==1]
        continents_list = all_continents['local'].unique()
        all_countries = filtered_df[filtered_df['tipo']==2]
        #countries_list = all_countries['local'].unique()
      
        countries = all_countries.melt(id_vars=['local'], value_vars=selected_columns, var_name='Ano_Mes', value_name='Valor')
        # CONVERTE A COLUNA ANO_MES PARA DATETIME E REMOVE O QUE NÃO CONSEGUIR CONVERTER (TOTAL)
        countries['Ano_Mes'] = countries['Ano_Mes'].apply(convert_to_datetime)
        countries = countries[countries['Ano_Mes'].notna()]
        countries = countries.set_index(['local','Ano_Mes'])
        totals = countries.groupby(level='local').sum()
        totals.sort_values(by='Valor', ascending=False, inplace=True)
        
        # Une aos dados de localização
        localized_countries = pd.DataFrame(totals).merge(geo_data, left_on='local', right_on='name')
        localized_countries.drop(columns=['country'], inplace=True)
        
        continents = all_continents.melt(id_vars=['local'], value_vars=selected_columns, var_name='Ano_Mes', value_name='Valor')
        continents = continents.set_index(['local','Ano_Mes'])
        continents =  continents.groupby(level='local').sum()
        continents.sort_values(by='Valor', ascending=False, inplace=True)
        
        
        # metricas gerais
        st.write("# Métricas Gerais")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        total_value = continents.sum()['Valor'].astype(int)
        first_continent = continents.iloc[0]['Valor'].astype(int)
        first_country = totals.iloc[0]['Valor'].astype(int)
        mean_continent = continents.mean()['Valor'].astype(int)
        mean_month = countries.groupby(level="Ano_Mes").sum().mean()[0].astype(int)
        mean_country = totals.mean()['Valor'].astype(int)

       
        col1.metric("Total de Viagens - " + selected, f'{total_value:,}'.replace(',','.'))
        col2.metric(f"Continente nº1 - {continents.index[0]}", f'{first_continent:,}'.replace(',','.') )
        col3.metric(f"País nº1 - {totals.index[0]}", f'{first_country:,}'.replace(',','.') )
        col4.metric("Média por Continentes", f'{mean_continent:,}'.replace(',','.') )
        col5.metric("Média por Mês", f'{mean_month:,}'.replace(',','.') )
        col6.metric("Média por País", f'{mean_country:,}'.replace(',','.') )

        
        st.write("# Mapa de Viagens por País")
        st.map(localized_countries, size='Valor', zoom=1)
        #st.write(localized_countries)
        
        selected_continent = st.selectbox("Selecione o Continente", continents_list)
        if selected_continent:
            #PREPARA DADOS PARA VISUALIZAÇÃO
            # filtra countries do continente selecionadp
            countries = all_countries[all_countries['continente']==selected_continent]
            # reestrutura o dataframe verticalizando os dados
            countries = countries.melt(id_vars=['local'], value_vars=selected_columns, var_name='Ano_Mes', value_name='Valor')
            #st.write(countries)

            # CONVERTE A COLUNA ANO_MES PARA DATETIME E REMOVE O QUE NÃO CONSEGUIR CONVERTER (TOTAL)
            countries['Ano_Mes'] = countries['Ano_Mes'].apply(convert_to_datetime)
            countries = countries[countries['Ano_Mes'].notna()]
            countries = countries.set_index(['local','Ano_Mes'])
            
            # Obtem os totais por pais
            totals = countries.groupby(level='local').sum()
            # Obtem os totalizados por paises e gera gráfico de pizza
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            totals.plot.pie(y='Valor', ax=ax1,  title=f'Total de Viagens por País - {selected_continent}')
            time_totals = countries.groupby(level='Ano_Mes').sum()
            time_totals.plot(kind="area", ax=ax2, title=f'Total de Viagens por Mês - {selected_continent}', xlabel='', ylabel='')
            st.write(fig)
            
            for (cat1, subdf) in countries.groupby(level='local'):
               
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                # não ta funcionando, pq?
                ax1.set_xticklabels([pandas_datetime.strftime("%m/%Y") for pandas_datetime in subdf.loc[cat1, :].index])
                ax2.set_xticklabels([pandas_datetime.strftime("%m/%Y") for pandas_datetime in subdf.loc[cat1, :].index])
                
                subdf.loc[cat1, :].plot(kind='bar', y='Valor', ax=ax1, rot=45, xlabel='', legend=False)
                subdf.loc[cat1, :].plot(kind='hist', y='Valor', ax=ax2, rot=45, xlabel='', legend=False)
                fig.suptitle(f'Viagens por Mês - {cat1}')
                fig.tight_layout()
                #plt.show()
                st.write(fig)
                #subdf.loc[cat1, :].plot.pie(y='Valor', ax=ax1, ylabel='', legend=False)
               
            #st.write(continentes.plot.bar(figsize=(10, 10)))
            

            #continentes.plot.pie(y="Valor",figsize=(10, 10))
        
       
        
        # rio_acumulados[[ 'lon', 'lat']] = rio_acumulados['centroide'].apply(pd.Series)
        # rio_acumulados = rio_acumulados.drop(columns=['index','id','nome','codarea', 'codmun','centroide'])
        # rio_acumulados['casosAcumulado'] = rio_acumulados['casosAcumulado'].astype(int) / 1000
        
        # st.write(" ### Casos Acumulados por Municipio no RJ - Em milhares")
        # st.map(rio_acumulados, size='casosAcumulado')
        # #st.write(rio_acumulados)
        
        #countries.plot.bar(x='local', y=selected_columns, rot=0)
   
    
   #https://github.com/google/dspl/blob/master/samples/google/canonical/countries.csv
  
    
    
    
    # #https://servicodados.ibge.gov.br/api/v2/malhas/33?resolucao=5
    # # get geojson from IBGE
    # url = 'https://servicodados.ibge.gov.br/api/v2/malhas/33?resolucao=5&formato=application/vnd.geo+json'
    # headers = {'Accept': 'application/vnd.geo+json'}
    # r = requests.get(url, headers=headers)
   
    # rio_de_janeiro =r.json()
    # #st.write(rio_de_janeiro)
    # rio_map_data = pd.DataFrame(rio_de_janeiro['features'])
    # rio_map_data = rio_map_data['properties'].apply(pd.Series)
    # rio_map_data.reset_index(inplace=True)
    # rio_map_data['codarea'] = pd.to_numeric(rio_map_data['codarea'])
    # #st.write(rio_map_data)
    
    
    
    # url = 'https://servicodados.ibge.gov.br/api/v1/localidades/estados/33/municipios'
    # headers = {'Accept': 'application/json'}
    # r = requests.get(url, headers=headers)
   
    # municipios_rio =r.json()
    # #st.write(municipios_rio)
    # municipios_rio = pd.DataFrame(municipios_rio).drop(columns=['microrregiao','regiao-imediata'])
    # #st.write(municipios_rio)
    
    # rio_municipios = pd.DataFrame(municipios_rio).merge(rio_map_data, left_on='id', right_on='codarea')
    # #st.write(rio_municipios)

    # rio_acumulados = data[(data.estado == 'RJ') & data.municipio.notnull()].groupby(['municipio','codmun']).sum().casosAcumulado.to_frame()
    # rio_acumulados.reset_index(inplace=True)
    # rio_acumulados['codmun'] = pd.to_numeric(rio_acumulados['codmun'])
    # #st.write(rio_acumulados)

    # rio_acumulados = pd.DataFrame(rio_acumulados).merge(rio_municipios, left_on='municipio', right_on='nome')
    # rio_acumulados[[ 'lon', 'lat']] = rio_acumulados['centroide'].apply(pd.Series)
    # rio_acumulados = rio_acumulados.drop(columns=['index','id','nome','codarea', 'codmun','centroide'])
    # rio_acumulados['casosAcumulado'] = rio_acumulados['casosAcumulado'].astype(int) / 1000
    
    # st.write(" ### Casos Acumulados por Municipio no RJ - Em milhares")
    # st.map(rio_acumulados, size='casosAcumulado')
    # #st.write(rio_acumulados)
    
    
    # regioes = data[data.regiao.isin(['Norte', 'Nordeste', 'Sudeste', 'Sul', 'Centro-Oeste']) & data.municipio.isna()].groupby(['regiao']).sum().reset_index()
    # regioes = regioes[['regiao', 'casosAcumulado']]

    # # Criando o gráfico de pizza
    # fig = px.pie(regioes, values='casosAcumulado', names='regiao', title='Distribuição Percentual dos Casos Acumulados de COVID-19 nas Regiões do Brasil')

    # # Customizando o gráfico
    # fig.update_traces(textposition='inside', textinfo='percent+label')

    # # Exibindo o gráfico
    # st.write(fig)
    
    
    
if __name__ == "__main__":
    Dashboard()


 